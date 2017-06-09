# cython: language_level=3
# cython: profile=False
# cython: binding=False
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
"""
# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""


import sys
import warnings
# noinspection PyUnresolvedReferences
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
# noinspection PyUnresolvedReferences
from libc.stdlib cimport strtol, strtof, strtod, malloc, free, realloc
# noinspection PyUnresolvedReferences
from libc.string cimport strcmp, memcpy
import numpy as np
cimport numpy as np
from cpython.ref cimport PyObject
cdef extern from "Python.h":
    char* PyByteArray_AS_STRING(object string)
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


#########################################################################################
# Constants


# for Windows compatibility
cdef double NAN = np.nan

# pre-define these characters for convenience and speed
cdef char TAB = b'\t'
cdef char LF = b'\n'
cdef char CR = b'\r'
cdef char HASH = b'#'
cdef char COLON = b':'
cdef char SEMICOLON = b';'
cdef char PERIOD = b'.'
cdef char COMMA = b','
cdef char SLASH = b'/'
cdef char PIPE = b'|'
cdef char EQUALS = b'='

# user field specifications for fixed fields
CHROM_FIELD = 'variants/CHROM'
POS_FIELD = 'variants/POS'
ID_FIELD = 'variants/ID'
REF_FIELD = 'variants/REF'
ALT_FIELD = 'variants/ALT'
QUAL_FIELD = 'variants/QUAL'


##########################################################################################
# Fused Types


ctypedef fused integer:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


ctypedef fused floating:
    # TODO float16?
    np.float32_t
    np.float64_t


##########################################################################################
# Vectors


cdef struct CharVector:
    int size
    int capacity
    char* data


cdef inline void CharVector_init(CharVector* self, int capacity) nogil:
    self.size = 0
    self.capacity = capacity
    self.data = <char*> malloc(sizeof(char) * capacity)


cdef inline void CharVector_free(CharVector* self) nogil:
    if self.data is not NULL:
        free(self.data)


cdef inline void CharVector_append(CharVector* self, char c) nogil:
    if self.size >= self.capacity:
        self.capacity *= 2
        self.data = <char*> realloc(self.data, sizeof(char) * self.capacity)
    self.data[self.size] = c
    self.size += 1


cdef inline void CharVector_clear(CharVector* self) nogil:
    self.size = 0


cdef inline void CharVector_terminate(CharVector* self) nogil:
    CharVector_append(self, 0)


cdef bytes CharVector_to_pybytes(CharVector* self):
    return PyBytes_FromStringAndSize(self.data, self.size)


cdef struct IntVector:
    int size
    int capacity
    int* data


cdef inline void IntVector_init(IntVector* self, int capacity) nogil:
    self.size = 0
    self.capacity = capacity
    self.data = <int*> malloc(sizeof(int) * capacity)


cdef inline void IntVector_free(IntVector* self) nogil:
    if self.data is not NULL:
        free(self.data)


cdef inline void IntVector_append(IntVector* self, int c) nogil:
    if self.size >= self.capacity:
        self.capacity *= 2
        self.data = <int*> realloc(self.data, sizeof(int) * self.capacity)
    self.data[self.size] = c
    self.size += 1


cdef inline void IntVector_clear(IntVector* self) nogil:
    self.size = 0


cdef inline void IntVector_terminate(IntVector* self) nogil:
    IntVector_append(self, 0)


##########################################################################################
# C string utilities


cdef inline int cstr_search_sorted(char* query, char** compare, int n_items) nogil:
    cdef:
        int i

    # TODO binary search

    # simple scan for now
    for i in range(n_items):
        if strcmp(query, compare[i]) == 0:
            return i

    return -1


##########################################################################################
# General I/O


cdef class InputStreamBase:
    """Abstract base class defining an input stream over C chars."""

    cdef:
        # character at the current position in the stream
        char c

    cdef int advance(self) nogil except -1:
        """Read the next character from the stream and store it in the `c` attribute."""
        pass


cdef class FileInputStream(InputStreamBase):

    cdef:
        # Python file-like object
        object fileobj
        int buffer_size
        bytearray buffer
        char* buffer_start
        char* buffer_end
        char* stream

    def __init__(self, fileobj, buffer_size=2**14):
        self.fileobj = fileobj
        self.buffer_size = buffer_size
        # initialise input buffer
        self.buffer = bytearray(buffer_size)
        self.buffer_start = PyByteArray_AS_STRING(self.buffer)
        self.stream = self.buffer_start
        self._bufferup()
        self.advance()

    cdef int _bufferup(self) nogil except -1:
        """Read as many bytes as possible from the underlying file object into the
        buffer."""
        cdef int l
        with gil:
            l = self.fileobj.readinto(self.buffer)
        if l > 0:
            self.stream = self.buffer_start
            self.buffer_end = self.buffer_start + l
        else:
            self.stream = NULL

    cdef int advance(self) nogil except -1:
        """Read the next character from the stream and store it in the `c` attribute."""
        if self.stream is self.buffer_end:
            self._bufferup()
        if self.stream is NULL:
            # end of file
            self.c = 0
        else:
            self.c = self.stream[0]
            self.stream += 1

    cdef int read_line_into(self, CharVector* dest) nogil except -1:
        """Read up to end of line or end of file (whichever comes first) and append
        chars to the `dest` buffer."""

        while True:

            if self.c == 0:
                break

            elif self.c == LF:
                CharVector_append(dest, LF)
                # advance input stream beyond EOL
                self.advance()
                break

            elif self.c == CR:
                # translate newdests
                CharVector_append(dest, LF)
                # advance input stream beyond EOL
                self.advance()
                if self.c == LF:
                    # handle Windows CRLF
                    self.advance()
                break

            else:
                CharVector_append(dest, self.c)
                self.advance()

    cdef int read_lines_into(self, CharVector* dest, int n) except -1:
        """Read up to `n` lines into the `dest` buffer."""
        cdef int n_lines_read = 0

        with nogil:

            while n_lines_read < n and self.c != 0:
                self.read_line_into(dest)
                n_lines_read += 1

        return n_lines_read

    # PYTHON API

    def readline(self):
        """Read characters up to end of line or end of file and return as Python bytes
        object."""
        cdef CharVector line
        CharVector_init(&line, 2**8)
        self.read_line_into(&line)
        ret = CharVector_to_pybytes(&line)
        CharVector_free(&line)
        return ret


cdef class CharVectorInputStream(InputStreamBase):

    cdef:
        CharVector vector
        int stream_index

    def __cinit__(self, int capacity):
        CharVector_init(&self.vector, capacity)
        self.stream_index = 0

    def __dealloc__(self):
        CharVector_free(&self.vector)

    cdef int advance(self) nogil except -1:
        if self.stream_index < self.vector.size:
            self.c = self.vector.data[self.stream_index]
            self.stream_index += 1
        else:
            self.c = 0

    cdef void clear(self) nogil:
        CharVector_clear(&self.vector)
        self.stream_index = 0


##########################################################################################
# VCF Parsing


cdef enum VCFState:
    CHROM,
    POS,
    ID,
    REF,
    ALT,
    QUAL,
    FILTER,
    INFO,
    FORMAT,
    CALLDATA,
    EOL,
    EOF


# cdef class VCFContext:
#
#     cdef:
#
#         # static attributes - should not change during parsing
#         int chunk_length
#         int n_samples
#
#         # dynamic attributes - reflect current state during parsing
#         int state  # overall parser state
#         int variant_index  # index of current variant
#         int chunk_variant_index  # index of current variant within current chunk
#         int sample_index  # index of current sample within call data
#         int sample_field_index  # index of field within call data for current sample
#         IntVector variant_format_indices  # indices of formats for the current variant
#
#         # buffers
#         CharVector temp  # used for numeric values
#         CharVector info_key  # used for info key
#         CharVector info_val  # used for info value
#
#         # keep track of current chrom and pos, even if fields are skipped
#         CharVector chrom
#         long pos
#
#     def __cinit__(self, int chunk_length, int n_samples):
#
#         # initialise static attributes
#         self.chunk_length = chunk_length
#         self.n_samples = n_samples
#
#         # initialise dynamic state
#         self.state = VCFState.CHROM
#         self.variant_index = -1
#         self.chunk_variant_index = -1
#         self.sample_index = 0
#         self.sample_field_index = 0
#         IntVector_init(&self.variant_format_indices, 2**6)
#
#         # initialise temporary buffers
#         CharVector_init(&self.temp, 2**6)
#         CharVector_init(&self.info_key, 2**6)
#         CharVector_init(&self.info_val, 2**6)
#
#         # initialise chrom and pos
#         CharVector_init(&self.chrom, 2**6)
#         self.pos = -1
#
#     def __dealloc__(self):
#         CharVector_free(&self.temp)
#         CharVector_free(&self.info_key)
#         CharVector_free(&self.info_val)
#         CharVector_free(&self.chrom)


cdef struct VCFContext:

    # static attributes - should not change during parsing
    int chunk_length
    int n_samples

    # dynamic attributes - reflect current state during parsing
    int state  # overall parser state
    int variant_index  # index of current variant
    int chunk_variant_index  # index of current variant within current chunk
    int sample_index  # index of current sample within call data
    int sample_field_index  # index of field within call data for current sample
    IntVector variant_format_indices  # indices of formats for the current variant

    # buffers
    CharVector temp  # used for numeric values
    CharVector info_key  # used for info key
    CharVector info_val  # used for info value

    # keep track of current chrom and pos, even if fields are skipped
    CharVector chrom
    long pos


cdef void VCFContext_init(VCFContext* self, int chunk_length, int n_samples):

    # initialise static attributes
    self.chunk_length = chunk_length
    self.n_samples = n_samples

    # initialise dynamic state
    self.state = VCFState.CHROM
    self.variant_index = -1
    self.chunk_variant_index = -1
    self.sample_index = 0
    self.sample_field_index = 0
    IntVector_init(&self.variant_format_indices, 2**6)

    # initialise temporary buffers
    CharVector_init(&self.temp, 2**6)
    CharVector_init(&self.info_key, 2**6)
    CharVector_init(&self.info_val, 2**6)

    # initialise chrom and pos
    CharVector_init(&self.chrom, 2**6)
    self.pos = -1


cdef void VCFContext_free(VCFContext* self):
    CharVector_free(&self.temp)
    CharVector_free(&self.info_key)
    CharVector_free(&self.info_val)
    CharVector_free(&self.chrom)


cdef class VCFChunkIterator:
    """TODO"""

    cdef:
        InputStreamBase stream
        VCFContext context
        VCFParser parser

    def __init__(self,
                 InputStreamBase stream,
                 int chunk_length,
                 headers,
                 fields,
                 types,
                 numbers):

        # store reference to input stream
        self.stream = stream

        # setup context
        n_samples = len(headers.samples)
        VCFContext_init(&self.context, chunk_length, n_samples)
        # self.context = VCFContext(chunk_length=chunk_length, n_samples=n_samples)

        # setup parser
        self.parser = VCFParser(fields=fields, types=types, numbers=numbers,
                                chunk_length=chunk_length, n_samples=n_samples)

    def __dealloc__(self):
        VCFContext_free(&self.context)

    def __iter__(self):
        return self

    def __next__(self):

        if self.context.state == VCFState.EOF:
            raise StopIteration

        # allocate arrays for next chunk
        self.parser.malloc_chunk()

        # parse next chunk
        self.parser.parse(self.stream, &self.context)

        # get the chunk
        chunk = self.parser.make_chunk(&self.context)

        if chunk is None:
            raise StopIteration

        return chunk


cdef class VCFParser:

    cdef:
        int chunk_length
        int n_samples
        VCFFieldParserBase chrom_parser
        VCFFieldParserBase pos_parser
        VCFFieldParserBase id_parser
        VCFFieldParserBase ref_parser
        VCFFieldParserBase alt_parser
        VCFFieldParserBase qual_parser
        VCFFieldParserBase filter_parser
        VCFFieldParserBase info_parser
        VCFFieldParserBase format_parser
        VCFFieldParserBase calldata_parser

    def __init__(self, fields, types, numbers, chunk_length, n_samples):
        self.chunk_length = chunk_length
        self.n_samples = n_samples

        # copy so we don't modify someone else's data
        fields = set(fields)

        # setup parsers
        self._init_chrom(fields, types)
        self._init_pos(fields, types)
        self._init_id(fields, types)
        self._init_ref(fields, types)
        self._init_alt(fields, types, numbers)
        self._init_qual(fields, types)
        self._init_filter(fields)
        self._init_info(fields, types, numbers)
        self._init_format_calldata(fields, types, numbers)

        if fields:
            # shouldn't ever be any left over
            raise RuntimeError('unexpected fields left over: %r' % set(fields))

    def _init_chrom(self, fields, types):
        """Setup CHROM parser."""
        if CHROM_FIELD in fields:
            chrom_parser = VCFChromParser(dtype=types[CHROM_FIELD], store=True,
                                          chunk_length=self.chunk_length)
            fields.remove(CHROM_FIELD)
        else:
            chrom_parser = VCFChromParser(dtype=None, store=False,
                                          chunk_length=self.chunk_length)
        chrom_parser.malloc_chunk()
        self.chrom_parser = chrom_parser

    def _init_pos(self, fields, types):
        """Setup POS parser."""
        if POS_FIELD in fields:
            if POS_FIELD in types:
                t = types[POS_FIELD]
                # TODO support user-provided type?
                if t != np.dtype('int32'):
                    warnings.warn('only int32 supported for POS field, ignoring '
                                  'requested type: %r' % types[POS_FIELD])
            pos_parser = VCFPosParser(store=True, chunk_length=self.chunk_length)
            fields.remove(POS_FIELD)
        else:
            pos_parser = VCFPosParser(store=False, chunk_length=self.chunk_length)
        pos_parser.malloc_chunk()
        self.pos_parser = pos_parser

    def _init_id(self, fields, types):
        """Setup ID parser."""
        if ID_FIELD in fields:
            id_parser = VCFStringFieldParser(key=b'ID', dtype=types[ID_FIELD],
                                             chunk_length=self.chunk_length)
            fields.remove(ID_FIELD)
        else:
            id_parser = VCFSkipFieldParser(key=b'ID')
        id_parser.malloc_chunk()
        self.id_parser = id_parser

    def _init_ref(self, fields, types):
        # setup REF parser
        if REF_FIELD in fields:
            ref_parser = VCFStringFieldParser(key=b'REF', dtype=types[REF_FIELD],
                                              chunk_length=self.chunk_length)
            fields.remove(REF_FIELD)
        else:
            ref_parser = VCFSkipFieldParser(key=b'REF')
        ref_parser.malloc_chunk()
        self.ref_parser = ref_parser

    def _init_alt(self, fields, types, numbers):
        """Setup ALT parser."""
        if ALT_FIELD in fields:
            t = types[ALT_FIELD]
            n = numbers[ALT_FIELD]
            alt_parser = VCFAltParser(dtype=t, number=n, chunk_length=self.chunk_length)
            fields.remove(ALT_FIELD)
        else:
            alt_parser = VCFSkipFieldParser(key=b'ALT')
        alt_parser.malloc_chunk()
        self.alt_parser = alt_parser

    def _init_qual(self, fields, types):
        """Setup QUAL parser."""
        if QUAL_FIELD in fields:
            if QUAL_FIELD in types:
                t = types[QUAL_FIELD]
                # TODO support user-provided type?
                if t != np.dtype('float32'):
                    warnings.warn('only float32 supported for QUAL field, ignoring '
                                  'requested type: %r' % types[QUAL_FIELD])
            qual_parser = VCFQualParser(chunk_length=self.chunk_length)
            fields.remove(QUAL_FIELD)
        else:
            qual_parser = VCFSkipFieldParser(key=b'QUAL')
        qual_parser.malloc_chunk()
        self.qual_parser = qual_parser

    def _init_filter(self, fields):
        """Setup FILTER parser."""
        filter_keys = list()
        for field in list(fields):
            if field.startswith('variants/FILTER_'):
                filter = field[16:].encode('ascii')
                filter_keys.append(filter)
                fields.remove(field)
        # debug(filter_keys, context)
        if filter_keys:
            filter_parser = VCFFilterParser(filters=filter_keys,
                                            chunk_length=self.chunk_length)
        else:
            filter_parser = VCFSkipFieldParser(key=b'FILTER')
        filter_parser.malloc_chunk()
        self.filter_parser = filter_parser

    def _init_info(self, fields, types, numbers):
        # setup INFO parser
        info_keys = list()
        info_types = dict()
        info_numbers = dict()
        # assume any variants fields left are INFO
        for field in list(fields):
            group, name = field.split('/')
            if group == 'variants':
                key = name.encode('ascii')
                info_keys.append(key)
                fields.remove(field)
                info_types[key] = types[field]
                info_numbers[key] = numbers[field]
        if info_keys:
            info_parser = VCFInfoParser(infos=info_keys, types=info_types,
                                        numbers=info_numbers,
                                        chunk_length=self.chunk_length)
        else:
            info_parser = VCFSkipFieldParser(key=b'INFO')
        info_parser.malloc_chunk()
        self.info_parser = info_parser

    def _init_format_calldata(self, fields, types, numbers):
        """Setup FORMAT and calldata parsers."""
        format_keys = list()
        format_types = dict()
        format_numbers = dict()
        for field in list(fields):
            group, name = field.split('/')
            if group == 'calldata':
                key = name.encode('ascii')
                format_keys.append(key)
                fields.remove(field)
                format_types[key] = types[field]
                format_numbers[key] = numbers[field]
        if format_keys:
            format_parser = VCFFormatParser(formats=format_keys)
            calldata_parser = VCFCallDataParser(formats=format_keys,
                                                types=format_types,
                                                numbers=format_numbers,
                                                chunk_length=self.chunk_length,
                                                n_samples=self.n_samples)
        else:
            format_parser = VCFSkipFieldParser(key=b'FORMAT')
            calldata_parser = VCFSkipAllCallDataParser()
        format_parser.malloc_chunk()
        calldata_parser.malloc_chunk()
        self.format_parser = format_parser
        self.calldata_parser = calldata_parser

    cdef int parse(self, InputStreamBase stream, VCFContext* context) except -1:
        """Parse to end of current chunk or EOF."""
        # debug('VCFParser.parse: enter', context)

        with nogil:

            while True:
                # debug('VCFParser.parse: loop; state %s; chunk_variant_index %s' % (
                #     context.state, context.chunk_variant_index,
                # ), context)

                if context.state == VCFState.EOF:
                    break

                elif context.state == VCFState.EOL:

                    # handle line terminators
                    if stream.c == LF:
                        stream.advance()
                    elif stream.c == CR:
                        stream.advance()
                        if stream.c == LF:
                            stream.advance()
                    else:
                        # shouldn't ever happen
                        warn('unexpected EOL character', context)
                        break

                    # advance state
                    context.state = VCFState.CHROM

                    # end of chunk?
                    if context.chunk_variant_index + 1 == context.chunk_length:
                        # we're done
                        break

                elif context.state == VCFState.CHROM:
                    self.chrom_parser.parse(stream, context)

                elif context.state == VCFState.POS:
                    self.pos_parser.parse(stream, context)

                elif context.state == VCFState.ID:
                    self.id_parser.parse(stream, context)

                elif context.state == VCFState.REF:
                    self.ref_parser.parse(stream, context)

                elif context.state == VCFState.ALT:
                    self.alt_parser.parse(stream, context)

                elif context.state == VCFState.QUAL:
                    self.qual_parser.parse(stream, context)

                elif context.state == VCFState.FILTER:
                    self.filter_parser.parse(stream, context)

                elif context.state == VCFState.INFO:
                    self.info_parser.parse(stream, context)

                elif context.state == VCFState.FORMAT:
                    self.format_parser.parse(stream, context)

                elif context.state == VCFState.CALLDATA:
                    self.calldata_parser.parse(stream, context)

                else:
                    # shouldn't ever happen
                    warn('unexpected parser state', context)
                    break

    cdef int malloc_chunk(self) except -1:
        self.chrom_parser.malloc_chunk()
        self.pos_parser.malloc_chunk()
        self.id_parser.malloc_chunk()
        self.ref_parser.malloc_chunk()
        self.alt_parser.malloc_chunk()
        self.qual_parser.malloc_chunk()
        self.filter_parser.malloc_chunk()
        self.info_parser.malloc_chunk()
        self.format_parser.malloc_chunk()
        self.calldata_parser.malloc_chunk()

    cdef object make_chunk(self, VCFContext* context):
        chunk_length = context.chunk_variant_index + 1
        if chunk_length > 0:
            if chunk_length < context.chunk_length:
                limit = chunk_length
            else:
                limit = None
            chunk = dict()
            self.chrom_parser.make_chunk(chunk, limit=limit)
            self.pos_parser.make_chunk(chunk, limit=limit)
            self.id_parser.make_chunk(chunk, limit=limit)
            self.ref_parser.make_chunk(chunk, limit=limit)
            self.alt_parser.make_chunk(chunk, limit=limit)
            self.qual_parser.make_chunk(chunk, limit=limit)
            self.filter_parser.make_chunk(chunk, limit=limit)
            self.info_parser.make_chunk(chunk, limit=limit)
            self.calldata_parser.make_chunk(chunk, limit=limit)
            context.chunk_variant_index = -1
            return chunk

        else:
            # TODO is this reachable?
            return None


cdef class VCFFieldParserBase:
    """Abstract base class for top-level field parsers."""

    cdef:
        bytes key
        np.dtype dtype
        int itemsize
        int number
        object fill
        int chunk_length
        np.ndarray values

    def __init__(self, key=None, dtype=None, number=1, fill=None, chunk_length=0):
        self.key = key
        if dtype is not None:
            dtype = np.dtype(dtype)
            self.itemsize = dtype.itemsize
        else:
            self.itemsize = 0
        self.dtype = dtype
        self.number = number
        self.fill = fill
        self.chunk_length = chunk_length

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        pass

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values
        if self.values.ndim > 1 and self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values[:limit]


cdef class VCFSkipFieldParser(VCFFieldParserBase):
    """Parser to skip a tab-delimited field."""

    def __init__(self, key):
        super(VCFSkipFieldParser, self).__init__(key=key)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                stream.advance()
                context.state += 1
                break

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


def check_string_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != 'S':
        raise ValueError('expected byte string ("S") dtype, found: %r' % dtype)
    return dtype


cdef class VCFChromParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.uint8_t[:] memory
        bint store

    def __init__(self, dtype, store, chunk_length):
        if store:
            dtype = check_string_dtype(dtype)
        super(VCFChromParser, self).__init__(key=b'CHROM', dtype=dtype, number=1,
                                             chunk_length=chunk_length)
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            # index into memory view
            int memory_index
            # number of characters read into current value
            int chars_stored = 0

        # setup context
        CharVector_clear(&context.chrom)
        context.pos = -1
        context.sample_index = 0
        context.sample_field_index = 0
        # check for EOF - important to handle file with no final line terminator
        if stream.c != 0:
            context.variant_index += 1
            context.chunk_variant_index += 1

        # now initialise memory index
        memory_index = context.chunk_variant_index * self.itemsize

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.advance()
                # advance to next field
                context.state += 1
                break

            else:

                # store on context
                CharVector_append(&context.chrom, stream.c)

                # store in chunk
                if self.store and chars_stored < self.itemsize:
                    # store value
                    self.memory[memory_index] = stream.c
                    # advance memory index
                    memory_index += 1
                    # advance number of characters stored
                    chars_stored += 1

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        if self.store:
            self.values = np.zeros(self.chunk_length, dtype=self.dtype)
            self.memory = self.values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store:
            chunk[CHROM_FIELD] = self.values[:limit]


cdef class VCFPosParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.int32_t[:] memory
        bint store

    def __init__(self, store, chunk_length):
        super(VCFPosParser, self).__init__(key=b'POS', dtype='int32', number=1,
                                           fill=-1, chunk_length=chunk_length)
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            long value
            int parsed

        # setup temp vector to store value
        CharVector_clear(&context.temp)

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                stream.advance()
                context.state += 1
                break

            else:
                CharVector_append(&context.temp, stream.c)

            # advance input stream
            stream.advance()

        # parse string as integer
        parsed = vcf_strtol(&context.temp, context, &value)

        # store value on context, whatever happens
        context.pos = value

        if parsed > 0 and self.store:
            # store value in chunk
            self.memory[context.chunk_variant_index] = value

    cdef int malloc_chunk(self) except -1:
        if self.store:
            self.values = np.zeros(self.chunk_length, dtype='int32')
            self.memory = self.values
            self.memory[:] = -1

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store:
            chunk[POS_FIELD] = self.values[:limit]


cdef class VCFStringFieldParser(VCFFieldParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, key, dtype, chunk_length):
        dtype = check_string_dtype(dtype)
        super(VCFStringFieldParser, self).__init__(key=key, dtype=dtype, number=1,
                                                   chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            # index into memory view
            int memory_index = context.chunk_variant_index * self.itemsize
            # number of characters read into current value
            int chars_stored = 0

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.advance()
                # advance to next field
                context.state += 1
                break

            elif chars_stored < self.itemsize:
                # store value
                self.memory[memory_index] = stream.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            stream.advance()

        return 1

    cdef int malloc_chunk(self) except -1:
        self.values = np.zeros(self.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')


cdef class VCFAltParser(VCFFieldParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, dtype, number, chunk_length):
        dtype = check_string_dtype(dtype)
        super(VCFAltParser, self).__init__(key=b'ALT', dtype=dtype, number=number,
                                           chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            # index of alt values
            int alt_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory offset and index
        memory_offset = context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            if stream.c == TAB:
                stream.advance()
                context.state += 1
                break

            elif stream.c == COMMA:
                # advance value index
                alt_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (alt_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            elif chars_stored < self.itemsize and alt_index < self.number:
                # store value
                self.memory[memory_index] = stream.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')


cdef class VCFQualParser(VCFFieldParserBase):
    """TODO"""

    cdef np.float32_t[:] memory

    def __init__(self, chunk_length):
        super(VCFQualParser, self).__init__(key=b'QUAL', dtype='float32', number=1,
                                            fill=NAN, chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            double value
            int parsed

        # reset temporary buffer
        CharVector_clear(&context.temp)

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.advance()
                context.state += 1
                break

            else:
                CharVector_append(&context.temp, stream.c)

            # advance input stream
            stream.advance()

        # parse string as floating
        parsed = vcf_strtod(&context.temp, context, &value)

        if parsed > 0:
            # store value
            self.memory[context.chunk_variant_index] = value

    cdef int malloc_chunk(self) except -1:
        self.values = np.empty(self.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = NAN


cdef class VCFFilterParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.uint8_t[:, :] memory
        tuple filters
        int n_filters
        char** filters_c

    def __cinit__(self, filters, chunk_length):
        self.filters = tuple(sorted(filters))
        self.n_filters = len(self.filters)
        self.filters_c = <char**> malloc(sizeof(char*) * self.n_filters)
        for i in range(self.n_filters):
            self.filters_c[i] = <char*> self.filters[i]

    def __init__(self, filters, chunk_length):
        super(VCFFilterParser, self).__init__(key=b'FILTER', dtype='bool',
                                              number=len(filters), fill=0,
                                              chunk_length=chunk_length)

    def __dealloc__(self):
        if self.filters_c is not NULL:
            free(self.filters_c)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            int filter_index

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return self.parse_missing(stream, context)

        # reset temporary buffer
        CharVector_clear(&context.temp)

        while True:

            if stream.c == 0:
                self.parse_filter(context)
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                self.parse_filter(context)
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                self.parse_filter(context)
                # advance input stream beyond tab
                stream.advance()
                context.state += 1
                break

            elif stream.c == COMMA or stream.c == COLON or stream.c == SEMICOLON:
                # some of these delimiters are not strictly kosher, but have seen them
                self.parse_filter(context)
                CharVector_clear(&context.temp)

            else:
                CharVector_append(&context.temp, stream.c)

            # advance input stream
            stream.advance()

        return 1

    cdef int parse_missing(self,
                           InputStreamBase stream,
                           VCFContext* context) nogil except -1:

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.advance()
                context.state += 1
                break

            # advance input stream
            stream.advance()

    cdef int parse_filter(self, VCFContext* context) nogil except -1:
        cdef:
            int filter_index
            int i
            char* f

        if context.temp.size == 0:
            warn('empty FILTER', context)
            return 0

        CharVector_terminate(&context.temp)

        # search through filters to find index
        filter_index = cstr_search_sorted(context.temp.data, self.filters_c,
                                          self.n_filters)

        # store value
        if filter_index >= 0:
            self.memory[context.chunk_variant_index, filter_index] = 1

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_filters)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]


##########################################################################################
# INFO parsing


cdef class VCFInfoParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        tuple infos
        int n_infos
        char** infos_c
        tuple info_parsers
        PyObject** info_parsers_c
        VCFInfoParserBase skip_parser

    def __cinit__(self, infos, types, numbers, chunk_length):

        # setup INFO keys
        self.infos = tuple(sorted(infos))
        self.n_infos = len(self.infos)

        # debug('infos: %s' % repr(self.infos))

        # setup INFO keys as C strings for nogil searching
        self.infos_c = <char**> malloc(sizeof(char*) * self.n_infos)
        for i in range(self.n_infos):
            self.infos_c[i] = <char*> self.infos[i]

        # setup INFO parsers
        info_parsers = list()
        self.skip_parser = VCFInfoSkipParser(key=None)
        for key in self.infos:
            t = types[key]
            n = numbers[key]
            if t == np.dtype(bool) or n == 0:
                parser = VCFInfoFlagParser(key, chunk_length=chunk_length)
            elif t == np.dtype('int32'):
                parser = VCFInfoInt32Parser(key, fill=-1, number=n, chunk_length=chunk_length)
            elif t == np.dtype('int64'):
                parser = VCFInfoInt64Parser(key, fill=-1, number=n, chunk_length=chunk_length)
            elif t == np.dtype('float32'):
                parser = VCFInfoFloat32Parser(key, fill=NAN, number=n, chunk_length=chunk_length)
            elif t == np.dtype('float64'):
                parser = VCFInfoFloat64Parser(key, fill=NAN, number=n, chunk_length=chunk_length)
            elif t == np.dtype(bool):
                parser = VCFInfoFlagParser(key, chunk_length=chunk_length)
            elif t.kind == 'S':
                parser = VCFInfoStringParser(key, dtype=t, number=n, chunk_length=chunk_length)
            else:
                parser = VCFInfoSkipParser(key)
                warnings.warn('type %s not supported for INFO field %r, field will be '
                              'skipped' % (t, key))
            info_parsers.append(parser)
        self.info_parsers = tuple(info_parsers)

        # store pointers to parsers for nogil trickery
        self.info_parsers_c = <PyObject**> malloc(sizeof(PyObject*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_parsers_c[i] = <PyObject*> self.info_parsers[i]

    def __init__(self, infos, types, numbers, chunk_length):
        super(VCFInfoParser, self).__init__(key=b'INFO', chunk_length=chunk_length)

    def __dealloc__(self):
        if self.infos_c is not NULL:
            free(self.infos_c)
        if self.info_parsers_c is not NULL:
            free(self.info_parsers_c)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return self.parse_missing(stream, context)

        # reset buffers
        CharVector_clear(&context.info_key)
        CharVector_clear(&context.info_val)

        while True:

            if stream.c == 0:
                if context.info_key.size > 0:
                    # handle flag
                    self.parse_info(stream, context)
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                if context.info_key.size > 0:
                    # handle flag
                    self.parse_info(stream, context)
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                if context.info_key.size > 0:
                    # handle flag
                    self.parse_info(stream, context)
                # advance input stream beyond tab
                stream.advance()
                context.state += 1
                break

            elif stream.c == SEMICOLON:
                if context.info_key.size > 0:
                    # handle flag
                    self.parse_info(stream, context)
                stream.advance()

            elif stream.c == EQUALS:
                # advance input stream beyond '='
                stream.advance()
                if context.info_key.size > 0:
                    self.parse_info(stream, context)
                else:
                    warn('missing INFO key', context)
                    self.skip_parser.parse(stream, context)

            else:

                CharVector_append(&context.info_key, stream.c)
                stream.advance()

    cdef int parse_missing(self,
                           InputStreamBase stream,
                           VCFContext* context) nogil except -1:

        # TODO refactor with filter

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.advance()
                context.state += 1
                break

            # advance input stream
            stream.advance()

    cdef int parse_info(self,
                        InputStreamBase stream,
                        VCFContext* context) nogil except -1:

        cdef:
            int parser_index
            PyObject* parser

        # terminate key
        CharVector_terminate(&context.info_key)

        # search for index of current INFO key
        parser_index = cstr_search_sorted(context.info_key.data, self.infos_c,
                                          self.n_infos)

        # clear out key for good measure
        CharVector_clear(&context.info_key)

        if parser_index >= 0:
            # obtain parser, use trickery for nogil
            parser = self.info_parsers_c[parser_index]
            (<VCFInfoParserBase> parser).parse(stream, context)
        else:
            self.skip_parser.parse(stream, context)

    cdef int malloc_chunk(self) except -1:
        cdef VCFInfoParserBase parser
        for parser in self.info_parsers:
            parser.malloc_chunk()

    cdef int make_chunk(self, chunk, limit=None) except -1:
        cdef VCFInfoParserBase parser
        for parser in self.info_parsers:
            parser.make_chunk(chunk, limit=limit)


cdef class VCFInfoParserBase:
    """TODO"""

    cdef:
        bytes key
        np.dtype dtype
        int itemsize
        int number
        object fill
        np.ndarray values
        int chunk_length

    def __init__(self, key=None, dtype=None, number=1, fill=0, chunk_length=0):
        self.key = key
        if dtype is not None:
            dtype = np.dtype(dtype)
            self.dtype = dtype
            self.itemsize = dtype.itemsize
        else:
            self.dtype = dtype
            self.itemsize = 0
        self.number = number
        self.fill = fill
        self.chunk_length = chunk_length

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values

    cdef int malloc_chunk(self) except -1:
        pass


cdef class VCFInfoInt32Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.int32_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        super(VCFInfoInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoInt64Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.int64_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        super(VCFInfoInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat32Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.float32_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        super(VCFInfoFloat32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_info_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat64Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.float64_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        super(VCFInfoFloat64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_info_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFlagParser(VCFInfoParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        super(VCFInfoFlagParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        # nothing to parse
        self.memory[context.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while stream.c != SEMICOLON and \
                stream.c != TAB and \
                stream.c != LF and \
                stream.c != CR and \
                stream.c != 0:
            stream.advance()

    cdef int make_chunk(self, chunk, limit=None) except -1:
        # override to view as bool array
        field = 'variants/' + str(self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)

    cdef int malloc_chunk(self) except -1:
        self.values = np.empty(self.chunk_length, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoStringParser(VCFInfoParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = check_string_dtype(kwargs.get('dtype'))
        super(VCFInfoStringParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # debug('InfoStringParser.parse', self.context)

        # initialise memory index
        memory_offset = context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        while True:

            if stream.c == 0 or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == TAB or \
                    stream.c == SEMICOLON:
                break

            elif stream.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = stream.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')


cdef class VCFInfoSkipParser(VCFInfoParserBase):
    """TODO"""

    def __init__(self, *args, **kwargs):
        super(VCFInfoSkipParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        # ensure we advance the end of the field
        while stream.c != SEMICOLON and \
                stream.c != TAB and \
                stream.c != LF and \
                stream.c != CR and \
                stream.c != 0:
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


cdef int vcf_info_parse_integer(InputStreamBase stream,
                                VCFContext* context,
                                integer[:, :] memory) nogil except-1:
    cdef:
        int value_index = 0

    # reset temporary buffer
    CharVector_clear(&context.info_val)

    while True:

        if stream.c == 0 or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == TAB or \
                stream.c == SEMICOLON:
            vcf_info_store_integer(context, value_index, memory)
            break

        elif stream.c == COMMA:
            vcf_info_store_integer(context, value_index, memory)
            CharVector_clear(&context.info_val)
            value_index += 1

        else:
            CharVector_append(&context.info_val, stream.c)

        stream.advance()


cdef int vcf_info_store_integer(VCFContext* context,
                                int value_index,
                                integer[:, :] memory) nogil except-1:
    cdef:
        int parsed
        long value

    if value_index >= memory.shape[1]:
        # more values than we have room for, ignore
        return 0

    # parse string as integer
    parsed = vcf_strtol(&context.info_val, context, &value)

    if parsed > 0:
        # store value
        memory[context.chunk_variant_index, value_index] = value


cdef int vcf_info_parse_floating(InputStreamBase stream,
                                 VCFContext* context,
                                 floating[:, :] memory) nogil except-1:
    cdef:
        int value_index = 0

    # reset temporary buffer
    CharVector_clear(&context.info_val)

    while True:

        if stream.c == 0 or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == TAB or \
                stream.c == SEMICOLON:
            vcf_info_store_floating(context, value_index, memory)
            break

        elif stream.c == COMMA:
            vcf_info_store_floating(context, value_index, memory)
            CharVector_clear(&context.info_val)
            value_index += 1

        else:
            CharVector_append(&context.info_val, stream.c)

        stream.advance()


cdef int vcf_info_store_floating(VCFContext* context,
                                 int value_index,
                                 floating[:, :] memory) nogil except-1:
    cdef:
        int parsed
        double value

    if value_index >= memory.shape[1]:
        # more values than we have room for, ignore
        return 0

    # parse string as floating
    parsed = vcf_strtod(&context.info_val, context, &value)

    if parsed > 0:
        # store value
        memory[context.chunk_variant_index, value_index] = value


##########################################################################################
# FORMAT and calldata parsing


cdef class VCFFormatParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        tuple formats
        int n_formats
        char** formats_c

    def __cinit__(self, formats):

        # setup FORMAT keys
        self.formats = tuple(sorted(formats))
        self.n_formats = len(self.formats)

        # setup FORMAT keys as C strings for nogil searching
        self.formats_c = <char**> malloc(sizeof(char*) * self.n_formats)
        for i in range(self.n_formats):
            self.formats_c[i] = <char*> self.formats[i]

        # debug('FORMATS: %s' % repr(self.formats))

    def __init__(self, formats):
        super(VCFFormatParser, self).__init__(key=b'FORMAT')

    def __dealloc__(self):
        if self.formats_c is not NULL:
            free(self.formats_c)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            int i

        # reset temporary buffer
        CharVector_clear(&context.temp)
        IntVector_clear(&context.variant_format_indices)

        while True:

            if stream.c == 0:
                # no point setting format, there will be no calldata
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                # no point setting format, there will be no calldata
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                self.store_format(context)
                # we're done here, advance to next field
                context.state += 1
                stream.advance()
                break

            elif stream.c == COLON:
                self.store_format(context)

            else:
                CharVector_append(&context.temp, stream.c)

            # advance to next character
            stream.advance()

    cdef int store_format(self, VCFContext* context) nogil except -1:
        cdef int format_index
        # debug('store_format: enter', context)

        # deal with empty or missing data
        if context.temp.size == 0:
            warn('empty FORMAT', context)
            return 0

        if context.temp.size == 1 and context.temp.data[0] == PERIOD:
            return 0

        # terminate the string
        CharVector_terminate(&context.temp)

        # find format index
        format_index = cstr_search_sorted(context.temp.data, self.formats_c,
                                          self.n_formats)

        # add to vector of indices for the current variant
        IntVector_append(&context.variant_format_indices, format_index)

        # clear out temp
        CharVector_clear(&context.temp)



cdef class VCFSkipAllCallDataParser(VCFFieldParserBase):
    """Skip a field."""

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            # advance input stream
            stream.advance()

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


cdef class VCFCallDataParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        tuple formats
        int n_formats
        tuple parsers
        PyObject** parsers_c
        VCFCallDataParserBase skip_parser
        int n_samples

    def __cinit__(self, formats, types, numbers, chunk_length, n_samples):
        self.chunk_length = chunk_length
        self.n_samples = n_samples

        # setup formats
        self.formats = tuple(sorted(formats))
        self.n_formats = len(self.formats)

        # setup parsers
        self.skip_parser = VCFCallDataSkipParser(key=None)
        parsers = list()
        kwds = dict(chunk_length=chunk_length, n_samples=n_samples)
        for key in self.formats:
            t = types[key]
            n = numbers[key]

            # special handling of "genotype" dtypes for any field
            if isinstance(t, str) and t.startswith('genotype/'):
                t = np.dtype(t.split('/')[1])
                if t == np.dtype('int8'):
                    parser = VCFGenotypeInt8Parser(key, number=n, fill=-1, **kwds)
                elif t == np.dtype('int16'):
                    parser = VCFGenotypeInt16Parser(key, number=n, fill=-1, **kwds)
                elif t == np.dtype('int32'):
                    parser = VCFGenotypeInt32Parser(key, number=n, fill=-1, **kwds)
                elif t == np.dtype('int64'):
                    parser = VCFGenotypeInt64Parser(key, number=n, fill=-1, **kwds)
                else:
                    warnings.warn('type %r not supported for genotype field %r, '
                                  'field will be skipped' % (t, key))
                    parser = self.skip_parser

            # special handling of GT field
            elif key == b'GT' and t == np.dtype('int8'):
                parser = VCFGenotypeInt8Parser(key, number=n, fill=-1, **kwds)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = VCFGenotypeInt16Parser(key, number=n, fill=-1, **kwds)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = VCFGenotypeInt32Parser(key, number=n, fill=-1, **kwds)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = VCFGenotypeInt64Parser(key, number=n, fill=-1, **kwds)

            # all other calldata
            elif t == np.dtype('int8'):
                parser = VCFCallDataInt8Parser(key, number=n, fill=-1, **kwds)
            elif t == np.dtype('int16'):
                parser = VCFCallDataInt16Parser(key, number=n, fill=-1, **kwds)
            elif t == np.dtype('int32'):
                parser = VCFCallDataInt32Parser(key, number=n, fill=-1, **kwds)
            elif t == np.dtype('int64'):
                parser = VCFCallDataInt64Parser(key, number=n, fill=-1, **kwds)
            elif t == np.dtype('float32'):
                parser = VCFCallDataFloat32Parser(key, number=n, fill=NAN, **kwds)
            elif t == np.dtype('float64'):
                parser = VCFCallDataFloat64Parser(key, number=n, fill=NAN, **kwds)
            elif t.kind == 'S':
                parser = VCFCallDataStringParser(key, dtype=t, number=n, **kwds)

            # TODO unsigned int parsers

            else:
                parser = VCFCallDataSkipParser(key)
                warnings.warn('type %r not supported for FORMAT field %r, field will be '
                              'skipped' % (t, key))

            parsers.append(parser)
        self.parsers = tuple(parsers)

        # store pointers to parsers
        self.parsers_c = <PyObject**> malloc(sizeof(PyObject*) * self.n_formats)
        for i in range(self.n_formats):
            self.parsers_c[i] = <PyObject*> self.parsers[i]

    def __init__(self, formats, types, numbers, chunk_length, n_samples):
        super(VCFCallDataParser, self).__init__(chunk_length=chunk_length)

    def __dealloc__(self):
        if self.parsers_c is not NULL:
            free(self.parsers_c)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        cdef:
            int i
            PyObject* parser

        # initialise context
        context.sample_index = 0
        context.sample_field_index = 0

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                context.sample_index += 1
                context.sample_field_index = 0
                stream.advance()

            elif stream.c == COLON:
                context.sample_field_index += 1
                stream.advance()

            elif context.sample_index >= context.n_samples:
                # more samples than we expected, skip to EOL
                while stream.c != 0 and stream.c != LF and stream.c != CR:
                    stream.advance()

            elif context.sample_field_index >= context.variant_format_indices.size:
                # more sample fields than formats declared for this variant
                self.skip_parser.parse(stream, context)

            else:

                # lookup format
                format_index = context.variant_format_indices.data[context.sample_field_index]

                # find parser
                if format_index >= 0:
                    parser = self.parsers_c[format_index]
                    # jump through some hoops to avoid references (which need the GIL)
                    (<VCFCallDataParserBase>parser).parse(stream, context)

                else:
                    self.skip_parser.parse(stream, context)

    cdef int malloc_chunk(self) except -1:
        cdef VCFCallDataParserBase parser
        for parser in self.parsers:
            parser.malloc_chunk()

    cdef int make_chunk(self, chunk, limit=None) except -1:
        cdef VCFCallDataParserBase parser
        for parser in self.parsers:
            parser.make_chunk(chunk, limit=limit)


cdef class VCFCallDataParserBase:

    cdef:
        bytes key
        np.dtype dtype
        int itemsize
        int number
        object fill
        np.ndarray values
        int chunk_length
        int n_samples

    def __init__(self, key=None, dtype=None, number=1, fill=0, chunk_length=0,
                 n_samples=0):
        self.key = key
        if dtype is not None:
            self.dtype = np.dtype(dtype)
            self.itemsize = self.dtype.itemsize
        else:
            self.dtype = dtype
            self.itemsize = 0
        self.number = number
        self.fill = fill
        self.chunk_length = chunk_length
        self.n_samples = n_samples

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        pass


cdef class VCFCallDataSkipParser(VCFCallDataParserBase):

    def __init__(self, key, *args, **kwargs):
        super(VCFCallDataSkipParser, self).__init__(key=key)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        while stream.c != COLON and \
                stream.c != TAB and \
                stream.c != CR and \
                stream.c != LF and \
                stream.c != 0:
            stream.advance()

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


cdef class VCFGenotypeInt8Parser(VCFCallDataParserBase):

    cdef:
        np.int8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        super(VCFGenotypeInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt16Parser(VCFCallDataParserBase):

    cdef:
        np.int16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        super(VCFGenotypeInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt32Parser(VCFCallDataParserBase):

    cdef:
        np.int32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        super(VCFGenotypeInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt64Parser(VCFCallDataParserBase):

    cdef:
        np.int64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        super(VCFGenotypeInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_genotype_parse(InputStreamBase stream,
                            VCFContext* context,
                            integer[:, :, :] memory) nogil except-1:
    cdef:
        int value_index = 0
    # debug('vcf_genotype_parse: enter', context)

    # reset temporary buffer
    CharVector_clear(&context.temp)

    while True:

        if stream.c == SLASH or stream.c == PIPE:
            vcf_genotype_store(context, memory, value_index)
            value_index += 1
            CharVector_clear(&context.temp)

        elif stream.c == COLON or \
                stream.c == TAB or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == 0:
            vcf_genotype_store(context, memory, value_index)
            break

        else:
            CharVector_append(&context.temp, stream.c)

        stream.advance()


cdef int vcf_genotype_store(VCFContext* context,
                            integer[:, :, :] memory,
                            int value_index) nogil except -1:
    cdef:
        int parsed
        long allele

    if value_index >= memory.shape[2]:
        # more values than we've made room for, ignore
        return 0

    # attempt to parse allele
    parsed = vcf_strtol(&context.temp, context, &allele)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_index, value_index] = allele


cdef class VCFCallDataInt8Parser(VCFCallDataParserBase):

    cdef np.int8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        super(VCFCallDataInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt16Parser(VCFCallDataParserBase):

    cdef np.int16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        super(VCFCallDataInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt32Parser(VCFCallDataParserBase):

    cdef np.int32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        super(VCFCallDataInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt64Parser(VCFCallDataParserBase):

    cdef np.int64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        super(VCFCallDataInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


# TODO unsigned int calldata parsers


cdef class VCFCallDataFloat32Parser(VCFCallDataParserBase):

    cdef np.float32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        super(VCFCallDataFloat32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataFloat64Parser(VCFCallDataParserBase):

    cdef np.float64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        super(VCFCallDataFloat64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext* context) nogil except -1:
        return vcf_calldata_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_calldata_parse_integer(InputStreamBase stream,
                                    VCFContext* context,
                                    integer[:, :, :] memory) nogil except-1:

    cdef:
        int value_index = 0
    # debug('vcf_calldata_parse_integer: enter', context)

    # reset temporary buffer
    CharVector_clear(&context.temp)

    while True:
        # debug('vcf_calldata_parse_integer: loop', context)

        if stream.c == COMMA:
            vcf_calldata_store_integer(context, value_index, memory)
            CharVector_clear(&context.temp)
            value_index += 1

        elif stream.c == COLON or \
                stream.c == TAB or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == 0:
            vcf_calldata_store_integer(context, value_index, memory)
            break

        else:
            CharVector_append(&context.temp, stream.c)

        stream.advance()


cdef int vcf_calldata_store_integer(VCFContext* context,
                                    int value_index,
                                    integer[:, :, :] memory) nogil except-1:
    cdef:
        int parsed
        long value

    if value_index >= memory.shape[2]:
        # more values than we have room for, ignore
        return 0

    parsed = vcf_strtol(&context.temp, context, &value)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_index, value_index] = value


cdef int vcf_calldata_parse_floating(InputStreamBase stream,
                                     VCFContext* context,
                                     floating[:, :, :] memory) nogil except-1:

    cdef:
        int value_index = 0
    # debug('vcf_calldata_parse_floating: enter', context)

    # reset temporary buffer
    CharVector_clear(&context.temp)

    while True:

        if stream.c == COMMA:
            vcf_calldata_store_floating(context, value_index, memory)
            CharVector_clear(&context.temp)
            value_index += 1

        elif stream.c == COLON or \
                stream.c == TAB or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == 0:
            vcf_calldata_store_floating(context, value_index, memory)
            break

        else:
            CharVector_append(&context.temp, stream.c)

        stream.advance()


cdef int vcf_calldata_store_floating(VCFContext* context,
                                     int value_index,
                                     floating[:, :, :] memory) nogil except-1:
    cdef:
        int parsed
        double value

    if value_index >= memory.shape[2]:
        # more values than we have room for, ignore
        return 0

    parsed = vcf_strtod(&context.temp, context, &value)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_index, value_index] = value


cdef class VCFCallDataStringParser(VCFCallDataParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = check_string_dtype(kwargs.get('dtype'))
        super(VCFCallDataStringParser, self).__init__(*args, **kwargs)

    cdef int parse(self,
                   InputStreamBase stream,
                   VCFContext* context) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # debug('CallDataStringParser.parse: enter', self.context)

        # initialise memory index
        memory_offset = ((context.chunk_variant_index *
                         context.n_samples *
                         self.number *
                         self.itemsize) +
                         (context.sample_index *
                          self.number *
                          self.itemsize))
        memory_index = memory_offset

        # read characters until tab
        while True:

            if stream.c == TAB or \
                    stream.c == COLON or \
                    stream.c == CR or \
                    stream.c == LF or \
                    stream.c == 0:
                return 1

            elif stream.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = stream.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'calldata/' + str(<bytes>self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values


##########################################################################################
# Low-level VCF value parsing functions


cdef int vcf_strtol(CharVector* value, VCFContext* context, long* l) nogil:
    cdef:
        char* str_end
        int parsed

    if value.size == 0:
        warn('expected integer, found empty value', context)
        return 0

    if value.size == 1 and value.data[0] == PERIOD:
        # explicit missing value
        return 0

    # terminate string
    CharVector_terminate(value)

    # do parsing
    l[0] = strtol(value.data, &str_end, 10)

    # check success
    parsed = str_end - value.data

    # check success
    if value.size - 1 == parsed:  # account for terminating null byte
        return parsed

    elif parsed > 0:
        warn('not all characters parsed for integer value', context)
        return parsed

    else:
        warn('error parsing integer value', context)
        return 0


cdef int vcf_strtod(CharVector* value, VCFContext* context, double* d) nogil:
    cdef:
        char* str_end
        int parsed

    if value.size == 0:
        warn('expected floating point number, found empty value', context)
        return 0

    if value.size == 1 and value.data[0] == PERIOD:
        # explicit missing value
        return 0

    # terminate string
    CharVector_terminate(value)

    # do parsing
    d[0] = strtod(value.data, &str_end)

    # check success
    parsed = str_end - value.data

    # check success
    if value.size - 1 == parsed:  # account for terminating null byte
        return parsed

    elif parsed > 0:
        warn('not all characters parsed for floating point value', context)
        return parsed

    else:
        warn('error parsing floating point value', context)
        return 0


##########################################################################################
# LOGGING


cdef int warn(message, VCFContext* context) nogil:
    with gil:
        # TODO customize message based on state (CHROM, POS, etc.)
        message += '; variant index: %s' % context.variant_index
        message += '; state: %s' % context.state
        message += '; temp: %s' % CharVector_to_pybytes(&context.temp)
        message += '; chunk_variant_index: %s' % context.chunk_variant_index
        message += '; sample_index: %s' % context.sample_index
        message += '; sample_field_index: %s' % context.sample_field_index
        message += '; chrom: %s' % CharVector_to_pybytes(&context.chrom)
        message += '; pos: %s' % context.pos
        warnings.warn(message)


cdef int debug(message, VCFContext* context) nogil except -1:
    with gil:
        message = '[DEBUG] ' + str(message) + '\n'
        # message += 'state: %s' % context.state
        # message += '; variant_index: %s' % context.variant_index
        # message += '; chunk_variant_index: %s' % context.chunk_variant_index
        # message += '; sample_index: %s' % context.sample_index
        # message += '; sample_field_index: %s' % context.sample_field_index
        # message += '; chrom: %s' % CharVector_to_pybytes(&context.chrom)
        # message += '; pos: %s' % context.pos
        print(message, file=sys.stderr)
        sys.stderr.flush()



##########################################################################################


import itertools


cdef class VCFChunkIteratorParallel:

    cdef:
        FileInputStream stream
        VCFContext* contexts
        VCFParser parser
        object pool
        int chunk_length
        int block_length
        int n_samples
        int n_threads
        list buffers
        int chunk_index

    def __cinit__(self,
                  FileInputStream stream,
                  int chunk_length, int block_length, int n_threads,
                  headers, fields, types, numbers):
        cdef int i
        self.stream = stream
        self.chunk_length = chunk_length
        self.block_length = block_length
        self.n_threads = n_threads
        self.pool = ThreadPool(n_threads)
        self.contexts = <VCFContext*> malloc(sizeof(VCFContext) * n_threads)
        self.n_samples = len(headers.samples)
        for i in range(n_threads):
            VCFContext_init(self.contexts + i, self.chunk_length, self.n_samples)
        self.buffers = [CharVectorInputStream(2**15) for _ in range(n_threads)]
        self.parser = VCFParser(fields=fields, types=types, numbers=numbers,
                                chunk_length=chunk_length, n_samples=self.n_samples)
        self.chunk_index = -1

    def __dealloc__(self):
        cdef int i
        if self.contexts is not NULL:
            for i in range(self.n_threads):
                VCFContext_free(self.contexts + i)
            free(self.contexts)

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            CharVectorInputStream buffer
            VCFContext* context = self.contexts
            int block_index = 0
            int i = 0
        self.chunk_index += 1
        # debug('__next__: enter %s' % self.chunk_index, context)

        # allocate arrays for next chunk
        self.parser.malloc_chunk()

        results = [None] * self.n_threads

        for i in itertools.cycle(list(range(self.n_threads))):
            context = self.contexts + i
            # debug('__next__: cycle %s, %s, %s' % (i, block_index, self.chunk_index),
            #       context)

            if results[i] is not None:
                # debug('wait for result', context)
                results[i].get()

            # debug('read into buffer - synchronous', context)
            buffer = self.buffers[i]
            buffer.clear()
            self.stream.read_lines_into(&buffer.vector, self.block_length)
            # tee up first character
            buffer.advance()
            # debug(CharVector_to_pybytes(&buffer.vector), context)

            # debug('do parsing - asynchronous', context)
            result = self.pool.apply_async(self.parse, args=(i, block_index,
                                                             self.chunk_index))
            results[i] = result

            # debug('increment block', context)
            block_index += 1

            # chunk done?
            if block_index * self.block_length >= self.chunk_length:
                # debug('chunk done', context)
                break

            # all done?
            if self.stream.c == 0:
                # debug('all done, EOF', context)
                # # TODO last chunk?
                # raise StopIteration
                break

        # debug('wait til all finished', context)
        for result in results:
            if result is not None:
                result.get()

        # debug('make chunk; %s; chunk_variant_index: %s' %
        #       (i, context.chunk_variant_index), context)
        chunk = self.parser.make_chunk(context)

        if chunk is None:
            self.pool.close()
            self.pool.join()
            self.pool.terminate()
            raise StopIteration
        else:
            # debug(chunk['variants/POS'], context)
            return chunk

    def parse(self, int i, block_index, chunk_index):
        cdef:
            VCFContext* context = self.contexts + i
            CharVectorInputStream buffer = self.buffers[i]
        # debug('async parse begin: %s, %s, %s' % (i, block_index, chunk_index), context)
        # debug('async parse: buffers: %s' % repr(self.buffers), context)
        context.state = VCFState.CHROM
        context.chunk_variant_index = block_index * self.block_length - 1
        context.variant_index = chunk_index * self.chunk_length + context.chunk_variant_index
        # debug('async parse: indices before: %s, %s' % (context.chunk_variant_index,
        #                                                context.variant_index),
        #       context)
        # debug('async parse: buffer: %r, %r, %r, %r' % (buffer,
        #                                                buffer.stream_index,
        #                                                <bytes>buffer.c,
        #                                                CharVector_to_pybytes(&buffer.vector)),
        #       context)
        self.parser.parse(buffer, context)
        # debug('async parse: indices after: %s, %s' % (context.chunk_variant_index,
        #                                               context.variant_index),
        #       context)
        # debug('async parse end: %s, %s, %s' % (i, block_index, chunk_index), context)
