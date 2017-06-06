# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: binding=False
# cython: linetrace=False

"""
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""


import sys
import warnings
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
# noinspection PyUnresolvedReferences
from libc.stdlib cimport strtol, strtof, strtod, malloc, free, realloc
from libc.string cimport strcmp, memcpy
import numpy as np
cimport numpy as np
# noinspection PyUnresolvedReferences
import cython
# noinspection PyUnresolvedReferences
cimport cython
from cpython.ref cimport PyObject
cdef extern from "Python.h":
    char* PyByteArray_AS_STRING(object string)


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


cdef class CharVector(object):
    """Dynamically-sized array of C chars."""

    cdef:
        int size
        int capacity
        char* data

    def __cinit__(self, capacity=16):
        self.size = 0
        self.capacity = capacity
        self.data = <char*> malloc(sizeof(char) * capacity)

    cdef void grow_if_full(self) nogil:
        """Double the capacity if the vector is full."""
        if self.size >= self.capacity:
            self.capacity *= 2
            self.data = <char*> realloc(self.data, sizeof(char) * self.capacity)

    cdef void append(self, char c) nogil:
        """Append a single char to the vector."""
        self.grow_if_full()
        self.data[self.size] = c
        self.size += 1

    cdef void clear(self) nogil:
        """Cheaply clear the vector by setting the size to 0."""
        self.size = 0

    cdef void terminate(self) nogil:
        """Terminate the vector by appending a null byte."""
        self.append(0)

    cdef bytes to_pybytes(self):
        return PyBytes_FromStringAndSize(self.data, self.size)


cdef class IntVector(object):
    """Dynamically-sized array of C ints."""

    cdef:
        int size
        int capacity
        int* data

    def __cinit__(self, capacity=16):
        self.size = 0
        self.capacity = capacity
        self.data = <int*> malloc(sizeof(int) * capacity)

    cdef void grow_if_full(self) nogil:
        """Double the capacity if the vector is full."""
        if self.size >= self.capacity:
            self.capacity *= 2
            self.data = <int*> realloc(self.data, sizeof(int) * self.capacity)

    cdef void append(self, int i) nogil:
        """Append a single value to the vector."""
        self.grow_if_full()
        self.data[self.size] = i
        self.size += 1

    cdef void clear(self) nogil:
        """Cheaply clear the vector by setting the size to 0."""
        self.size = 0


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


cdef class InputStreamBase(object):
    """Abstract base class defining an input stream over C chars."""

    cdef:
        # character at the current position in the stream
        char c

    cdef int getc(self) nogil except -1:
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

    def __cinit__(self, fileobj, buffer_size=2**14):
        self.fileobj = fileobj
        self.buffer_size = buffer_size
        # initialise input buffer
        self.buffer = bytearray(buffer_size)
        self.buffer_start = PyByteArray_AS_STRING(self.buffer)
        self.stream = self.buffer_start
        self._bufferup()
        self.getc()

    # CYTHON API: PRIVATE

    cdef int _bufferup(self) nogil except -1:
        """Read as many bytes as possible from the underlying file object into the
        buffer."""
        cdef int l
        with gil:
            self.fileobj.readinto(self.buffer)
        if l > 0:
            self.stream = self.buffer_start
            self.buffer_end = self.buffer_start + l
        else:
            self.stream = NULL

    # CYTHON API: PUBLIC

    cdef int getc(self) nogil except -1:
        """Read the next character from the stream and store it in the `c` attribute."""
        if self.stream is self.buffer_end:
            self._bufferup()
        if self.stream is NULL:
            # end of file
            self.c = 0
        else:
            self.c = self.stream[0]
            self.stream += 1

    cdef int read_line_into(self, CharVector dest) nogil except -1:
        """Read up to end of line or end of file (whichever comes first) and append
        chars to the `dest` buffer."""

        while True:

            if self.c == 0:
                break

            elif self.c == LF:
                dest.append(LF)
                # advance input stream beyond EOL
                self.getc()
                break

            elif self.c == CR:
                # translate newdests
                dest.append(LF)
                # advance input stream beyond EOL
                self.getc()
                if self.c == LF:
                    # handle Windows CRLF
                    self.getc()
                break

            else:
                dest.append(self.c)
                self.getc()

    cdef int read_lines_into(self, CharVector dest, int n) nogil except -1:
        """Read up to `n` lines into the `dest` buffer."""
        cdef int n_lines_read = 0

        while n_lines_read < n and self.c != 0:
            self.read_line_into(dest)
            n_lines_read += 1

        return n_lines_read

    # PYTHON API

    def readline(self):
        """Read characters up to end of line or end of file and return as Python bytes
        object."""
        cdef CharVector line = CharVector()
        self.read_line_into(line)
        return line.to_pybytes()


cdef class CharVectorInputStream(InputStreamBase):

    cdef:
        CharVector vector
        int stream_index
        int vector_size

    def __cinit__(self, CharVector vector):
        self.vector = vector
        self.stream_index = 0
        self.getc()

    cdef int getc(self) nogil except -1:
        if self.stream_index < self.vector.size:
            self.c = self.vector.data[self.stream_index]
            self.stream_index += 1
        else:
            self.c = 0


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


cdef class VCFContext(object):

    cdef:

        # static attributes - should not change during parsing
        int chunk_length
        int n_samples
        int ploidy

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

    def __cinit__(self, int chunk_length, int n_samples, int ploidy):

        # initialise static attributes
        self.chunk_length = chunk_length
        self.n_samples = n_samples
        self.ploidy = ploidy

        # initialise dynamic state
        self.state = VCFState.CHROM
        self.variant_index = -1
        self.chunk_variant_index = -1
        self.sample_index = 0
        self.sample_field_index = 0
        self.variant_format_indices = IntVector()

        # initialise temporary buffers
        self.temp = CharVector()
        self.info_key = CharVector()
        self.info_val = CharVector()

        # initialise chrom and pos
        self.chrom = CharVector()
        self.pos = -1


cdef class VCFChunkIterator(object):
    """TODO"""

    cdef:
        InputStreamBase stream
        VCFContext context
        VCFChunkParser parser

    def __init__(self,
                 InputStreamBase stream,
                 int chunk_length,
                 headers,
                 fields,
                 types,
                 numbers,
                 ploidy=2):

        # store reference to input stream
        self.stream = stream

        # setup context
        n_samples = len(headers.samples)
        self.context = VCFContext(chunk_length=chunk_length, n_samples=n_samples,
                                  ploidy=ploidy)

        # setup parser
        self.parser = VCFChunkParser(fields=fields, types=types, numbers=numbers)

    def __iter__(self):
        return self

    def __next__(self):

        # allocate arrays for next chunk
        self.parser.malloc_chunk(self.context)

        # parse next chunk
        self.parser.parse(self.stream, self.context)

        # get the chunk
        chunk = self.parser.make_chunk()

        return chunk


cdef class VCFChunkParser(object):

    cdef:
        VCFFieldParserBase chrom_parser
        VCFFieldParserBase pos_parser
        VCFFieldParserBase id_parser
        VCFFieldParserBase ref_parser
        VCFFieldParserBase alt_parser
        VCFFieldParserBase qual_parser
        VCFFieldParserBase filter_parser
        VCFFieldParserBase format_parser
        VCFFieldParserBase calldata_parser

    def __init__(self, fields, types, numbers):

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
        self._init_info(self, fields, types, numbers)
        self._init_format_calldata(self, fields, types, numbers)

        if fields:
            # shouldn't ever be any left over
            raise RuntimeError('unexpected fields left over: %r' % set(fields))

    def _init_chrom(self, fields, types, numbers):
        """Setup CHROM parser."""
        if CHROM_FIELD in fields:
            chrom_parser = VCFChromParser(dtype=types[CHROM_FIELD], skip=False)
            fields.remove(CHROM_FIELD)
        else:
            chrom_parser = VCFChromParser(dtype=types[CHROM_FIELD], skip=True)
        chrom_parser.malloc_chunk()
        self.chrom_parser = chrom_parser

    def _init_pos(self, fields, types):
        """Setup POS parser."""
        if POS_FIELD in fields:
            if POS_FIELD in types:
                # TODO support user-provided type?
                warn('Only int32 supported for POS field, ignoring requested type: %r'
                     % types[POS_FIELD])
            pos_parser = VCFPosParser(skip=False)
            fields.remove(POS_FIELD)
        else:
            pos_parser = VCFPosParser(skip=True)
        pos_parser.malloc_chunk()
        self.pos_parser = pos_parser

    def _init_id(self, fields, types):
        """Setup ID parser."""
        if ID_FIELD in fields:
            id_parser = VCFStringFieldParser(field=ID_FIELD, dtype=types[ID_FIELD])
            fields.remove(ID_FIELD)
        else:
            id_parser = VCFSkipFieldParser()
        id_parser.malloc_chunk()
        self.id_parser = id_parser

    def _init_ref(self, fields, types):
        # setup REF parser
        if REF_FIELD in fields:
            ref_parser = VCFStringFieldParser(field=REF_FIELD, dtype=types[REF_FIELD])
            fields.remove(REF_FIELD)
        else:
            ref_parser = VCFSkipFieldParser()
        ref_parser.malloc_chunk()
        self.ref_parser = ref_parser

    def _init_alt(self, fields, types, numbers):
        """Setup ALT parser."""
        if ALT_FIELD in fields:
            t = types[ALT_FIELD]
            n = numbers[ALT_FIELD]
            alt_parser = VCFAltParser(dtype=t, number=n)
            fields.remove(ALT_FIELD)
        else:
            alt_parser = VCFSkipFieldParser()
        alt_parser.malloc_chunk()
        self.alt_parser = alt_parser

    def _init_qual(self, fields, types):
        """Setup QUAL parser."""
        if QUAL_FIELD in fields:
            if QUAL_FIELD in types:
                # TODO support user-provided type?
                warn('Only float32 supported for QUAL field, ignoring requested type: %r'
                     % types[QUAL_FIELD])
            qual_parser = VCFQualParser()
            fields.remove(QUAL_FIELD)
        else:
            qual_parser = VCFSkipFieldParser()
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
            filter_parser = VCFFilterParser(filters=filter_keys)
        else:
            filter_parser = VCFSkipFieldParser()
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
                                        numbers=info_numbers)
        else:
            info_parser = VCFSkipFieldParser()
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
            format_parser = VCFFormatParser()
            calldata_parser = VCFCallsParser(formats=format_keys,
                                             types=format_types,
                                             numbers=format_numbers)
        else:
            format_parser = VCFSkipFieldParser()
            calldata_parser = VCFSkipAllCalldataParser()
        format_parser.malloc_chunk()
        calldata_parser.malloc_chunk()
        self.format_parser = format_parser
        self.calldata_parser = calldata_parser


    def parse(self, InputStreamBase stream, VCFContext context):
        """Parse to end of current chunk or EOF."""

        with nogil:

            while True:

                if context.state == VCFState.EOF:
                    break

                elif context.state == VCFState.EOL:

                    # handle line terminators
                    if stream.c == LF:
                        stream.getc()
                    elif stream.c == CR:
                        stream.getc()
                        if stream.c == LF:
                            stream.getc()
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

    def malloc_chunk(self, VCFContext context):
        self.chrom_parser.malloc_chunk(context)
        self.pos_parser.malloc_chunk(context)
        self.id_parser.malloc_chunk(context)
        self.ref_parser.malloc_chunk(context)
        self.alt_parser.malloc_chunk(context)
        self.qual_parser.malloc_chunk(context)
        self.filter_parser.malloc_chunk(context)
        self.format_parser.malloc_chunk(context)
        self.calldata_parser.malloc_chunk(context)

    def make_chunk(self, VCFContext context):
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


cdef class VCFFieldParserBase(object):
    """Abstract base class for top-level field parsers."""

    cdef:
        char* key
        int number
        object values
        object fill
        object dtype
        int itemsize

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context):
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFSkipFieldParser(VCFFieldParserBase):
    """Parser to skip a tab-delimited field."""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                stream.getc()
                context.state += 1
                break

            # advance input stream
            stream.getc()

    def malloc_chunk(self, VCFContext context):
        pass

    def make_chunk(self, chunk, limit=None):
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

    def __init__(self, dtype, bint store):
        if store:
            self.dtype = check_string_dtype(dtype)
            self.itemsize = self.dtype.itemsize
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_chrom_parse(stream, context, self.memory, self.itemsize, self.store)

    def malloc_chunk(self, VCFContext context):
        if self.store:
            self.values = np.zeros(context.chunk_length, dtype=self.dtype)
            self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        if self.store:
            chunk[CHROM_FIELD] = self.values[:limit]


cdef int vcf_chrom_parse(InputStreamBase stream,
                         VCFContext context,
                         np.uint8_t[:] memory,
                         int itemsize,
                         bint store) nogil except -1:
    cdef:
        # index into memory view
        int memory_index = context.chunk_variant_index * itemsize
        # number of characters read into current value
        int chars_stored = 0

    # setup to store CHROM on context
    context.chrom.clear()

    # check for EOF - important to handle file with no final line terminator
    if stream.c != 0:
        context.variant_index += 1
        context.chunk_variant_index += 1

    while True:

        if stream.c == 0:
            context.state = VCFState.EOF
            return 0

        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            return 0

        elif stream.c == TAB:
            # advance input stream beyond tab
            stream.getc()
            # advance to next field
            context.state += 1
            return 1

        else:

            # store on context
            context.chrom.append(stream.c)

            # store in chunk
            if store and chars_stored < itemsize:
                # store value
                memory[memory_index] = stream.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

        # advance input stream
        stream.getc()


cdef class VCFPosParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.int32_t[:] memory
        bint store

    def __init__(self, bint store):
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_pos_parse(stream, context, self.memory, self.store)

    def malloc_chunk(self, VCFContext context):
        if self.store:
            self.values = np.zeros(context.chunk_length, dtype='int32')
            self.memory = self.values
            self.memory[:] = -1

    def make_chunk(self, chunk, limit=None):
        if self.store:
            chunk[POS_FIELD] = self.values[:limit]


cdef int vcf_pos_parse(InputStreamBase stream,
                       VCFContext context,
                       integer[:] memory,
                       bint store) nogil except -1:
    cdef:
        long value

    # setup temp vector to store value
    context.temp.clear()

    while True:

        if stream.c == 0:
            context.state = VCFState.EOF
            break

        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            break

        elif stream.c == TAB:
            stream.getc()
            context.state += 1
            break

        else:
            context.temp.append(stream.c)

        # advance input stream
        stream.getc()

    # parse string as integer
    value = vcf_strtol(context.temp, -1, context)

    # store value on context
    context.pos = value

    # store value in chunk
    if store:
        memory[context.chunk_variant_index] = value


cdef class VCFStringFieldParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.uint8_t[:] memory
        object field

    def __init__(self, field, dtype):
        self.field = field
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_string_field_parse(stream, context, self.memory, self.itemsize)

    def malloc_chunk(self, VCFContext context):
        self.values = np.zeros(context.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]


cdef int vcf_string_field_parse(InputStreamBase stream,
                                VCFContext context,
                                np.uint8_t[:] memory,
                                int itemsize) nogil except -1:
    cdef:
        # index into memory view
        int memory_index = context.chunk_variant_index * itemsize
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
            stream.getc()
            # advance to next field
            context.state += 1
            break

        elif chars_stored < itemsize:
            # store value
            memory[memory_index] = stream.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1

        # advance input stream
        stream.getc()

    return 1


cdef class VCFAltParser(VCFFieldParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, dtype, number):
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_alt_parse(stream, context, self.memory, self.itemsize, self.number)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    def make_chunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values


cdef int vcf_alt_parse(InputStreamBase stream,
                       VCFContext context,
                       np.uint8_t[:] memory,
                       int itemsize,
                       int number) nogil except -1:
    cdef:
        # index of alt values
        int alt_index = 0
        # index into memory view
        int memory_offset, memory_index
        # number of characters read into current value
        int chars_stored = 0

    # initialise memory offset and index
    memory_offset = context.chunk_variant_index * itemsize * number
    memory_index = memory_offset

    while True:

        if stream.c == 0:
            context.state = VCFState.EOF
            break

        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            break

        if stream.c == TAB:
            stream.getc()
            context.state += 1
            break

        elif stream.c == COMMA:
            # advance value index
            alt_index += 1
            # set memory index to beginning of next item
            memory_index = memory_offset + (alt_index * itemsize)
            # reset chars stored
            chars_stored = 0

        elif chars_stored < itemsize and alt_index < number:
            # store value
            memory[memory_index] = stream.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1

        # advance input stream
        stream.getc()


cdef class VCFQualParser(VCFFieldParserBase):
    """TODO"""

    cdef np.float32_t[:] memory

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_qual_parse(stream, context, self.memory)

    def malloc_chunk(self, VCFContext context):
        self.values = np.empty(context.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = NAN

    def make_chunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]


cdef int vcf_qual_parse(InputStreamBase stream,
                        VCFContext context,
                        floating[:] memory) nogil except -1:
    cdef:
        double value

    # reset temporary buffer
    context.temp.clear()

    while True:

        if stream.c == 0:
            context.state = VCFState.EOF
            break

        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            break

        elif stream.c == TAB:
            # advance input stream beyond tab
            stream.getc()
            context.state += 1
            break

        else:
            context.temp.append(stream.c)

        # advance input stream
        stream.getc()

    # parse string as floating
    value = vcf_strtol(context.temp, NAN, context)

    # store value
    memory[context.chunk_variant_index] = value


cdef class VCFFilterParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.uint8_t[:, :] memory
        tuple filters
        int n_filters
        char** filter_ptrs

    def __cinit__(self, filters):
        self.filters = tuple(sorted(filters))
        self.n_filters = len(self.filters)
        self.filter_ptrs = <char**> malloc(sizeof(char*) * self.n_filters)
        for i in range(self.n_filters):
            self.filter_ptrs[i] = <char*> self.filters[i]

    def __dealloc__(self):
        if self.filter_ptrs is not NULL:
            free(self.filter_ptrs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_filter_parse(stream, context, self.filter_ptrs, self.n_filters,
                                self.memory)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.n_filters)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]


cdef int vcf_filter_parse(InputStreamBase stream,
                          VCFContext context,
                          char** filter_ptrs,
                          int n_filters,
                          np.uint8_t[:, :] memory) nogil except -1:
    cdef:
        int filter_index

    # reset temporary buffer
    context.temp.clear()

    # check for explicit missing value
    if stream.c == PERIOD:
        # treat leading period as missing, regardless of what comes next

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                # advance input stream beyond tab
                stream.getc()
                context.state += 1
                break

            # advance input stream
            stream.getc()

        # bail out early
        return 0

    while True:

        if stream.c == 0:
            vcf_filter_store(context, filter_ptrs, n_filters, memory)
            context.state = VCFState.EOF
            break

        elif stream.c == LF or stream.c == CR:
            vcf_filter_store(context, filter_ptrs, n_filters, memory)
            context.state = VCFState.EOL
            break

        elif stream.c == TAB:
            vcf_filter_store(context, filter_ptrs, n_filters, memory)
            # advance input stream beyond tab
            stream.getc()
            context.state += 1
            break

        elif stream.c == COMMA or stream.c == COLON or stream.c == SEMICOLON:
            # some of these delimiters are not strictly kosher, but have seen them
            vcf_filter_store(context, filter_ptrs, n_filters, memory)
            context.temp.clear()

        else:
            context.temp.append(stream.c)

        # advance input stream
        stream.getc()

    return 1


cdef int vcf_filter_store(VCFContext context,
                          char** filter_ptrs,
                          int n_filters,
                          np.uint8_t[:, :] memory) nogil except -1:
    cdef:
        int filter_index
        int i
        char* f

    if context.temp.size == 0:
        warn('empty FILTER', context)
        return 0

    context.temp.terminate()

    # search through filters to find index
    filter_index = cstr_search_sorted(context.temp, filter_ptrs, n_filters)

    # store value
    if filter_index >= 0:
        memory[context.chunk_variant_index, filter_index] = 1


cdef class VCFInfoParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        tuple infos
        int n_infos
        char** info_ptrs
        tuple info_parsers
        PyObject** info_parser_ptrs
        VCFInfoParserBase skip_parser


    def __cinit__(self, infos, types, numbers):

        # setup INFO keys
        self.infos = tuple(sorted(infos))
        self.n_infos = len(self.infos)

        # setup INFO keys as C strings for nogil searching
        self.info_ptrs = <char**> malloc(sizeof(char*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_ptrs[i] = <char*> self.infos[i]

        # setup INFO parsers
        info_parsers = list()
        self.skip_parser = VCFInfoSkipParser()
        for key in self.infos:
            t = types[key]
            n = numbers[key]
            if t == np.dtype(bool) or n == 0:
                parser = VCFInfoFlagParser(key)
            elif t == np.dtype('int32'):
                parser = VCFInfoInt32Parser(key, fill=-1, number=n)
            elif t == np.dtype('int64'):
                parser = VCFInfoInt64Parser(key, fill=-1, number=n)
            elif t == np.dtype('float32'):
                parser = VCFInfoFloat32Parser(key, fill=NAN, number=n)
            elif t == np.dtype('float64'):
                parser = VCFInfoFloat64Parser(key, fill=NAN, number=n)
            elif t == np.dtype(bool):
                parser = VCFInfoFlagParser(key)
            elif t.kind == 'S':
                parser = VCFInfoStringParser(key, dtype=t, number=n)
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for INFO field %r, field will be '
                              'skipped' % (t, key))
            info_parsers.append(parser)
        self.info_parsers = tuple(info_parsers)

        # store pointers to parsers for nogil trickery
        self.info_parser_ptrs = <PyObject**> malloc(sizeof(PyObject*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_parser_ptrs[i] = <PyObject*> self.info_parsers[i]

    def __dealloc__(self):
        if self.info_ptrs is not NULL:
            free(self.info_ptrs)
        if self.info_parser_ptrs is not NULL:
            free(self.info_parser_ptrs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse(stream, context)

    def malloc_chunk(self, VCFContext context):
        for parser in self.info_parsers:
            parser.malloc_chunk()

    def make_chunk(self, chunk, limit=None):
        for parser in self.info_parsers:
            parser.make_chunk(chunk, limit=limit)




cdef class VCFInfoParserBase(object):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoInt32Parser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoInt64Parser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoFloat32Parser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoFloat64Parser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoFlagParser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoStringParser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFInfoSkipParser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFFormatParser(object):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFCallsParser(object):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def malloc_chunk(self, VCFContext context)::
        pass

    def make_chunk(self, chunk, limit=None):
        pass


##########################################################################################
# Low-level VCF value parsing functions


cdef long vcf_strtol(CharVector value, long default, VCFContext context) nogil:
    cdef:
        char* str_end
        int parsed
        long l

    if value.size == 0:
        warn('expected integer, found empty value', context)
        return default

    if value.size == 1 and value.data[0] == PERIOD:
        # explicit missing value
        return default

    # terminate string
    value.terminate()

    # do parsing
    l = strtol(value.data, &str_end, 10)

    # check success
    parsed = str_end - value.data

    # check success
    if value.size - 1 == parsed:  # account for terminating null byte
        return l

    elif parsed > 0:
        warn('not all characters parsed for integer value', context)
        return l

    else:
        warn('error parsing integer value', context)
        return default


cdef double vcf_strtod(CharVector value, double default, VCFContext context) nogil:
    cdef:
        char* str_end
        int parsed
        double d

    if value.size == 0:
        warn('expected floating point number, found empty value', context)
        return default

    if value.size == 1 and value.data[0] == PERIOD:
        # explicit missing value
        return default

    # terminate string
    value.terminate()

    # do parsing
    l = strtod(value.data, &str_end)

    # check success
    parsed = str_end - value.data

    # check success
    if value.size - 1 == parsed:  # account for terminating null byte
        return l

    elif parsed > 0:
        warn('not all characters parsed for floating point value', context)
        return l

    else:
        warn('error parsing floating point value', context)
        return default




















##########################################################################################
# LOGGING


cdef int warn(message, VCFContext context) nogil:
    with gil:
        # TODO customize message based on state (CHROM, POS, etc.)
        message += '; variant index: %s' % context.variant_index
        warnings.warn(message)


cdef int debug(msg, VCFContext context=None) nogil except -1:
    with gil:
        msg = '[DEBUG] ' + str(msg) + '\n'
        if context is not None:
            msg += 'state: %s' % context.state
            msg += '; variant_index: %s' % context.variant_index
            msg += '; chunk_variant_index: %s' % context.chunk_variant_index
            msg += '; sample_index: %s' % context.sample_index
        print(msg, file=sys.stderr)
        sys.stderr.flush()


##########################################################################################
# LEGACY


# cdef class ParserContext:
#     cdef:
#         # input file and buffer
#         object input_file
#         int input_buffer_size
#         bytearray input_buffer
#         char* input
#         char* input_start
#         char* input_end
#         # temporary buffer
#         int temp_buffer_size
#         bytearray temp_buffer
#         char* temp
#         int temp_size
#         # state
#         int state
#         char c
#         long l
#         double d
#         int n_samples
#         int variant_index
#         int chunk_variant_index
#         int sample_index
#         int chunk_length
#         int ploidy
#         # infos
#         int n_infos
#         tuple infos
#         char** info_ptrs
#         list info_parsers
#         PyObject** info_parser_ptrs
#         # filters
#         int n_filters
#         tuple filters
#         char** filter_ptrs
#         # formats
#         int n_formats
#         tuple formats
#         char** format_ptrs
#         int variant_n_formats
#         int format_index
#         list calldata_parsers
#         PyObject** calldata_parser_ptrs
#         PyObject** variant_calldata_parser_ptrs
#
#
#     def __cinit__(self,
#                   input_file,
#                   int input_buffer_size,
#                   int temp_buffer_size,
#                   int n_samples,
#                   int chunk_length,
#                   int ploidy):
#
#         # initialize input buffer
#         self.input_file = input_file
#         self.input_buffer_size = input_buffer_size
#         self.input_buffer = bytearray(input_buffer_size)
#         self.input_start = PyByteArray_AS_STRING(self.input_buffer)
#         self.input = self.input_start
#         context_fill_buffer(self)
#         context_getc(self)
#
#         # initialize temporary buffer
#         self.temp_buffer = bytearray(temp_buffer_size)
#         self.temp = PyByteArray_AS_STRING(self.temp_buffer)
#         self.temp_size = 0
#
#         # initialize pointer arrays
#         self.variant_calldata_parser_ptrs = <PyObject**> malloc(MAX_FORMATS *
#                                                                 sizeof(PyObject*))
#
#         # initialize state
#         self.state = VCFState.CHROM
#         self.n_samples = n_samples
#         self.variant_index = -1
#         self.chunk_variant_index = -1
#         self.sample_index = 0
#         self.format_index = 0
#         self.chunk_length = chunk_length
#         self.ploidy = ploidy
#
#     def __dealloc__(self):
#
#         # infos
#         if self.info_ptrs is not NULL:
#             free(self.info_ptrs)
#         if self.info_parser_ptrs is not NULL:
#             free(self.info_parser_ptrs)
#
#         # filters
#         if self.filter_ptrs is NULL:
#             free(self.filter_ptrs)
#
#         # calldata
#         if self.format_ptrs is not NULL:
#             free(self.format_ptrs)
#         if self.calldata_parser_ptrs is not NULL:
#             free(self.calldata_parser_ptrs)
#         if self.variant_calldata_parser_ptrs is not NULL:
#             free(self.variant_calldata_parser_ptrs)
#
#
# cdef class SkipAllCalldataParser(Parser):
#     """Skip a field."""
#
#     cdef int parse(self) nogil except -1:
#         return skip_all_calldata(self.context)
#
#
# cdef inline int skip_all_calldata(ParserContext context) nogil except -1:
#
#     while True:
#
#         if stream.c == 0:
#             context.state = VCFState.EOF
#             return 1
#
#         elif stream.c == LF or stream.c == CR:
#             context.state = VCFState.EOL
#             return 1
#
#         # advance input stream
#         stream.getc()
#
#
# cdef inline int info_parse(InfoParser self,
#                            ParserContext context) nogil except -1:
#
#
#     # debug('info_parse', context)
#
#     # check for explicit missing value
#     if stream.c == PERIOD:
#
#         while True:
#
#             if stream.c == 0:
#                 context.state = VCFState.EOF
#                 return 0
#
#             elif stream.c == LF or stream.c == CR:
#                 context.state = VCFState.EOL
#                 return 0
#
#             elif stream.c == TAB:
#                 # advance input stream beyond tab
#                 stream.getc()
#                 context.state += 1
#                 return 1
#
#             # advance input stream
#             stream.getc()
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == 0:
#             info_store(self, context)
#             context.state = VCFState.EOF
#             return 0
#
#         elif stream.c == LF or stream.c == CR:
#             info_store(self, context)
#             context.state = VCFState.EOL
#             return 0
#
#         elif stream.c == TAB:
#             info_store(self, context)
#             # advance input stream beyond tab
#             stream.getc()
#             context.state += 1
#             return 1
#
#         elif stream.c == SEMICOLON:
#             info_store(self, context)
#             stream.getc()
#
#         elif stream.c == EQUALS:
#             # advance input stream beyond '='
#             # TODO warn if empty key?
#             stream.getc()
#             info_store(self, context)
#
#         else:
#
#             temp_append(context)
#             stream.getc()
#
#
# cdef inline int info_store(InfoParser self,
#                            ParserContext context) nogil except -1:
#     cdef:
#         int parser_index
#         PyObject* parser
#
#     if context.temp_size > 0:
#         # treat temp as key
#
#         # terminate temp
#         context.temp.terminate()
#
#         # search for index of current INFO key
#         parser_index = search_sorted(context.temp, context.info_ptrs, context.n_infos)
#
#         # clear out temp for good measure
#         context.temp.clear()
#
#         if parser_index >= 0:
#             # obtain parser
#             parser = context.info_parser_ptrs[parser_index]
#             (<Parser> parser).parse()
#         else:
#             self.skip_parser.parse()
#
#     # else:
#     #
#     #     warn('error parsing INFO field, missing key', context)
#     #     # advance to next delimiter
#     #     while stream.c != SEMICOLON and \
#     #             stream.c != TAB and \
#     #             stream.c != LF and \
#     #             stream.c != CR and \
#     #             stream.c != 0:
#     #         stream.getc()
#     #
#     return 1
#
#
# cdef class InfoParserBase(Parser):
#
#     def __init__(self, ParserContext context, key, fill, number):
#         super(InfoParserBase, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#         self.fill = fill
#         self.number = number
#
#     def mkchunk(self, chunk, limit=None):
#         field = 'variants/' + str(<bytes>self.key, 'ascii')
#         values = self.values[:limit]
#         if self.number == 1:
#             values = values.squeeze(axis=1)
#         chunk[field] = values
#         self.malloc()
#
#
# cdef class InfoInt32Parser(InfoParserBase):
#
#     cdef np.int32_t[:, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return info_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.number)
#         self.values = np.empty(shape, dtype='int32')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class InfoInt64Parser(InfoParserBase):
#
#     cdef np.int64_t[:, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return info_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.number)
#         self.values = np.empty(shape, dtype='int64')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef inline int info_integer_parse(integer[:, :] memory,
#                                    int number,
#                                    ParserContext context) nogil except -1:
#     cdef:
#         int value_index = 0
#
#     # debug('info_integer_parse', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == 0 or \
#                 stream.c == LF or \
#                 stream.c == CR or \
#                 stream.c == TAB or \
#                 stream.c == SEMICOLON:
#             info_integer_store(memory, number, context, value_index)
#             break
#
#         elif stream.c == COMMA:
#             info_integer_store(memory, number, context, value_index)
#             context.temp.clear()
#             value_index += 1
#
#         else:
#             temp_append(context)
#
#         stream.getc()
#
#     # reset temporary buffer here to indicate new field
#     context.temp.clear()
#
#     return 1
#
#
# cdef inline int info_integer_store(integer[:, :] memory,
#                                    int number,
#                                    ParserContext context,
#                                    int value_index) nogil except -1:
#     cdef:
#         int success
#
#     if value_index >= number:
#         # more values than we have room for, ignore
#         return 1
#
#     # parse string as integer
#     success = temp_tolong(context)
#
#     # store value
#     if success:
#         memory[context.chunk_variant_index, value_index] = context.l
#
#     return 1
#
#
# cdef class InfoFloat32Parser(InfoParserBase):
#
#     cdef np.float32_t[:, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return info_floating_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.number)
#         self.values = np.empty(shape, dtype='float32')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class InfoFloat64Parser(InfoParserBase):
#
#     cdef np.float64_t[:, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return info_floating_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.number)
#         self.values = np.empty(shape, dtype='float64')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef inline int info_floating_parse(floating[:, :] memory,
#                                     int number,
#                                     ParserContext context) nogil except -1:
#     cdef:
#         int value_index = 0
#
#     # debug('info_floating_parse', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == 0 or \
#                 stream.c == LF or \
#                 stream.c == CR or \
#                 stream.c == TAB or \
#                 stream.c == SEMICOLON:
#             info_floating_store(memory, number, context, value_index)
#             break
#
#         elif stream.c == COMMA:
#             info_floating_store(memory, number, context, value_index)
#             context.temp.clear()
#             value_index += 1
#
#         else:
#             temp_append(context)
#
#         stream.getc()
#
#     # reset temporary buffer here to indicate new field
#     context.temp.clear()
#
#     return 1
#
#
# cdef inline int info_floating_store(floating[:, :] memory,
#                                     int number,
#                                     ParserContext context,
#                                     int value_index) nogil except -1:
#     cdef:
#         int success
#
#     if value_index >= number:
#         # more values than we have room for, ignore
#         return 1
#
#     # parse string as double
#     success = temp_todouble(context)
#
#     # store value
#     if success:
#         memory[context.chunk_variant_index, value_index] = context.d
#
#     return 1
#
#
# cdef class InfoFlagParser(Parser):
#
#     cdef np.uint8_t[:] memory
#
#     def __init__(self, ParserContext context, key):
#         super(InfoFlagParser, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#
#     cdef int parse(self) nogil except -1:
#         # debug('InfoFlagParser.parse', self.context)
#         self.memory[self.context.chunk_variant_index] = 1
#         # ensure we advance the end of the field
#         while self.stream.c != SEMICOLON and \
#                 self.stream.c != TAB and \
#                 self.stream.c != LF and \
#                 self.stream.c != CR and \
#                 self.stream.c != 0:
#             context_getc(self.context)
#         return 1
#
#     def malloc(self):
#         self.values = np.zeros(self.context.chunk_length, dtype='u1')
#         self.memory = self.values
#
#     def mkchunk(self, chunk, limit=None):
#         field = 'variants/' + str(<bytes>self.key, 'ascii')
#         chunk[field] = self.values[:limit].view(bool)
#         self.malloc()
#
#
# cdef class InfoStringParser(Parser):
#
#     cdef np.uint8_t[:] memory
#
#     def __init__(self, ParserContext context, key, dtype, number):
#         super(InfoStringParser, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#         self.dtype = check_string_dtype(dtype)
#         self.itemsize = self.dtype.itemsize
#         self.number = number
#
#     cdef int parse(self) nogil except -1:
#         cdef:
#             int value_index = 0
#             # index into memory view
#             int memory_offset, memory_index
#             # number of characters read into current value
#             int chars_stored = 0
#
#         # debug('InfoStringParser.parse', self.context)
#
#         # initialise memory index
#         memory_offset = self.context.chunk_variant_index * self.itemsize * self.number
#         memory_index = memory_offset
#
#         while True:
#
#             if self.stream.c == 0 or \
#                     self.stream.c == LF or \
#                     self.stream.c == CR or \
#                     self.stream.c == TAB or \
#                     self.stream.c == SEMICOLON:
#                 break
#
#             elif self.stream.c == COMMA:
#                 # advance value index
#                 value_index += 1
#                 # set memory index to beginning of next item
#                 memory_index = memory_offset + (value_index * self.itemsize)
#                 # reset chars stored
#                 chars_stored = 0
#
#             elif chars_stored < self.itemsize and value_index < self.number:
#                 # store value
#                 self.memory[memory_index] = self.stream.c
#                 # advance memory index
#                 memory_index += 1
#                 # advance number of characters stored
#                 chars_stored += 1
#
#             # advance input stream
#             context_getc(self.context)
#
#         return 1
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.number)
#         self.values = np.zeros(shape, dtype=self.dtype)
#         self.memory = self.values.reshape(-1).view('u1')
#
#     def mkchunk(self, chunk, limit=None):
#         field = 'variants/' + str(self.key, 'ascii')
#         values = self.values[:limit]
#         if self.number == 1:
#             values = values.squeeze(axis=1)
#         chunk[field] = values
#         self.malloc()
#
#
# cdef class FormatParser(Parser):
#
#     cdef int parse(self) nogil except -1:
#         return format_parse(self, self.context)
#
#
# # break out method as function for profiling
# cdef inline int format_parse(FormatParser self,
#                              ParserContext context) nogil except -1:
#     cdef:
#         int format_index = 0
#         int i
#
#     # debug('format_parse: enter', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#     context.variant_n_formats = 0
#
#     while True:
#
#         if stream.c == 0:
#             # no point setting format
#             context.state = VCFState.EOF
#             return 0
#
#         elif stream.c == LF or stream.c == CR:
#             # no point setting format
#             context.state = VCFState.EOL
#             return 0
#
#         elif stream.c == TAB:
#             # debug('format_parse: field end, setting', context)
#             format_set(context, format_index)
#             context.variant_n_formats = format_index + 1
#             # we're done here, advance to next field
#             context.state += 1
#             stream.getc()
#             return 1
#
#         elif stream.c == COLON:
#             # debug('format_parse: format end, setting', context)
#             format_set(context, format_index)
#             format_index += 1
#
#         else:
#             temp_append(context)
#
#         # advance to next character
#         stream.getc()
#
#
# cdef inline int format_set(ParserContext context,
#                            int format_index) nogil except -1:
#     cdef:
#         PyObject* parser = NULL
#         int parser_index
#     # debug('format_set: enter', context)
#
#     if format_index >= MAX_FORMATS:
#         warn('MAX_FORMATS exceeded', context)
#         return 0
#
#     if context.temp_size == 0:
#         warn('empty FORMAT', context)
#
#     else:
#
#         # terminate string
#         context.temp.terminate()
#
#         # find parser
#         parser_index = search_sorted(context.temp, context.format_ptrs, context.n_formats)
#
#         # set parser
#         if parser_index >= 0:
#             parser = context.calldata_parser_ptrs[parser_index]
#
#     context.variant_calldata_parser_ptrs[format_index] = parser
#
#     context.temp.clear()
#
#     return 1
#
#
# # noinspection PyShadowingBuiltins
# cdef class CalldataParser(Parser):
#
#     cdef tuple formats
#     cdef Parser skip_parser
#
#     def __init__(self, ParserContext context, formats, types, numbers):
#         super(CalldataParser, self).__init__(context)
#         self.skip_parser = SkipCalldataFieldParser(context)
#
#         # setup formats
#         context.formats = tuple(sorted(formats))
#         context.n_formats = len(context.formats)
#         if context.format_ptrs is not NULL:
#             free(context.format_ptrs)
#         context.format_ptrs = <char**> malloc(context.n_formats * sizeof(char*))
#         for i in range(context.n_formats):
#             context.format_ptrs[i] = <char*> context.formats[i]
#
#         # setup parsers
#         context.calldata_parsers = list()
#         for key in context.formats:
#             t = types[key]
#             n = numbers[key]
#             if key == b'GT' and t == np.dtype('int8'):
#                 parser = GenotypeInt8Parser(context, key, fill=-1)
#             elif key == b'GT' and t == np.dtype('int16'):
#                 parser = GenotypeInt16Parser(context, key, fill=-1)
#             elif key == b'GT' and t == np.dtype('int32'):
#                 parser = GenotypeInt32Parser(context, key, fill=-1)
#             elif key == b'GT' and t == np.dtype('int64'):
#                 parser = GenotypeInt64Parser(context, key, fill=-1)
#             elif t == np.dtype('int8'):
#                 parser = CalldataInt8Parser(context, key, number=n, fill=-1)
#             elif t == np.dtype('int16'):
#                 parser = CalldataInt16Parser(context, key, number=n, fill=-1)
#             elif t == np.dtype('int32'):
#                 parser = CalldataInt32Parser(context, key, number=n, fill=-1)
#             elif t == np.dtype('int64'):
#                 parser = CalldataInt64Parser(context, key, number=n, fill=-1)
#             elif t == np.dtype('float32'):
#                 parser = CalldataFloat32Parser(context, key, number=n, fill=NAN)
#             elif t == np.dtype('float64'):
#                 parser = CalldataFloat64Parser(context, key, number=n, fill=NAN)
#             elif t.kind == 'S':
#                 parser = CalldataStringParser(context, key, dtype=t, number=n)
#             # TODO unsigned int parsers
#             else:
#                 parser = self.skip_parser
#                 warnings.warn('type %s not supported for FORMAT field %r, field will be '
#                               'skipped' % (t, key))
#             context.calldata_parsers.append(parser)
#
#         # store pointers for
#         if context.calldata_parser_ptrs is not NULL:
#             free(context.calldata_parser_ptrs)
#         context.calldata_parser_ptrs = <PyObject**> malloc(context.n_formats *
#                                                            sizeof(PyObject*))
#         for i in range(context.n_formats):
#             context.calldata_parser_ptrs[i] = <PyObject*> context.calldata_parsers[i]
#
#     cdef int parse(self) nogil except -1:
#         return calldata_parse(self, self.context)
#
#     def malloc(self):
#         cdef Parser parser
#         for parser in self.context.calldata_parsers:
#             parser.malloc()
#
#     def mkchunk(self, chunk, limit=None):
#         cdef Parser parser
#         for parser in self.context.calldata_parsers:
#             parser.mkchunk(chunk, limit=limit)
#
#
# # break out method as function for profiling
# cdef inline int calldata_parse(CalldataParser self,
#                                ParserContext context) nogil except -1:
#     cdef:
#         int i
#         PyObject* parser
#
#     # debug('calldata_parse: enter', context)
#
#     # initialise context
#     context.sample_index = 0
#     context.format_index = 0
#
#     while True:
#
#         if stream.c == 0:
#             # debug('calldata_parse: EOF', context)
#             context.state = VCFState.EOF
#             return 1
#
#         elif stream.c == LF or stream.c == CR:
#             # debug('calldata_parse: EOL', context)
#             context.state = VCFState.EOL
#             context.sample_index = 0
#             return 1
#
#         elif stream.c == TAB:
#             # debug('calldata_parse: TAB', context)
#             context.sample_index += 1
#             context.format_index = 0
#             stream.getc()
#
#         elif stream.c == COLON:
#             # debug('calldata_parse: COLON', context)
#             context.format_index += 1
#             stream.getc()
#
#         elif context.format_index < MAX_FORMATS:
#             # debug('calldata_parse: parse', context)
#
#             parser = context.variant_calldata_parser_ptrs[context.format_index]
#             if parser is NULL:
#                 self.skip_parser.parse()
#
#             else:
#                 # jump through some hoops to avoid references (which need the GIL)
#                 (<Parser>parser).parse()
#
#         else:
#             # debug('calldata_parse: exceeded MAX_FORMATS', context)
#             self.skip_parser.parse()
#
#
# cdef class SkipInfoFieldParser(Parser):
#
#     cdef int parse(self) nogil except -1:
#         while self.stream.c != SEMICOLON and \
#                 self.stream.c != TAB and \
#                 self.stream.c != CR and \
#                 self.stream.c != LF and \
#                 self.stream.c != 0:
#             context_getc(self.context)
#         return 1
#
#
# cdef class SkipCalldataFieldParser(Parser):
#
#     cdef int parse(self) nogil except -1:
#         while self.stream.c != COLON and \
#                 self.stream.c != TAB and \
#                 self.stream.c != CR and \
#                 self.stream.c != LF and \
#                 self.stream.c != 0:
#             context_getc(self.context)
#         return 1
#
#
# cdef inline int calldata_integer_parse(integer[:, :, :] memory,
#                                        int number,
#                                        ParserContext context) nogil except -1:
#     cdef:
#         int value_index = 0
#
#     # debug('calldata_integer_parse: enter', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == COMMA:
#             # debug('calldata_integer_parse: COMMA', context)
#             calldata_integer_store(memory, number, context, value_index)
#             context.temp.clear()
#             value_index += 1
#
#         elif stream.c == COLON or \
#                 stream.c == TAB or \
#                 stream.c == LF or \
#                 stream.c == CR or \
#                 stream.c == 0:
#             # debug('calldata_integer_parse: DELIMITER', context)
#             calldata_integer_store(memory, number, context, value_index)
#             break
#
#         else:
#             # debug('calldata_integer_parse: temp_append', context)
#             temp_append(context)
#
#         # debug('calldata_integer_parse: getc', context)
#         stream.getc()
#
#     # debug('calldata_integer_parse: exit', context)
#     return 1
#
#
# cdef inline int calldata_integer_store(integer[:, :, :] memory,
#                                        int number,
#                                        ParserContext context,
#                                        int value_index) nogil except -1:
#     cdef:
#         int success
#
#     # debug('calldata_integer_store: enter', context)
#
#     if value_index >= number:
#         # more values than we have room for, ignore
#         return 1
#
#     # debug('parse string as integer', context)
#     success = temp_tolong(context)
#
#     # store value
#     if success:
#         memory[context.chunk_variant_index, context.sample_index, value_index] = context.l
#
#     # debug('calldata_integer_store: exit', context)
#     return 1
#
#
# cdef inline int calldata_floating_parse(floating[:, :, :] memory,
#                                         int number,
#                                         ParserContext context) nogil except -1:
#     cdef:
#         int value_index = 0
#
#     # debug('calldata_floating_parse: enter', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == COMMA:
#             calldata_floating_store(memory, number, context, value_index)
#             context.temp.clear()
#             value_index += 1
#
#         elif stream.c == COLON or \
#                 stream.c == TAB or \
#                 stream.c == LF or \
#                 stream.c == CR or \
#                 stream.c == 0:
#             calldata_floating_store(memory, number, context, value_index)
#             return 1
#
#         else:
#             temp_append(context)
#
#         stream.getc()
#
#
# cdef inline int calldata_floating_store(floating[:, :, :] memory,
#                                         int number,
#                                         ParserContext context,
#                                         int value_index) nogil except -1:
#     cdef:
#         int success
#
#     if value_index >= number:
#         # more values than we have room for, ignore
#         return 1
#
#     # parse string as floating
#     success = temp_todouble(context)
#
#     # store value
#     if success:
#         memory[context.chunk_variant_index, context.sample_index, value_index] = context.d
#
#     return 1
#
#
# cdef class GenotypeParserBase(Parser):
#
#     def __init__(self, ParserContext context, bytes key, fill):
#         # debug('GenotypeParserBase.__init__: enter', context)
#         super(GenotypeParserBase, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#         self.fill = fill
#
#     def mkchunk(self, chunk, limit=None):
#         chunk['calldata/GT'] = self.values[:limit]
#         self.malloc()
#
#
# cdef class GenotypeInt8Parser(GenotypeParserBase):
#
#     cdef np.int8_t[:, :, :] memory
#     # TODO cdef object dtype = 'int8' ... can factor out malloc?
#
#     def malloc(self):
#         # debug('GenotypeInt8Parser.malloc: enter', self.context)
#         shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
#         self.values = np.empty(shape, dtype='int8')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#     cdef int parse(self) nogil except -1:
#         # debug('GenotypeInt8Parser.parse: enter', self.context)
#         return genotype_parse(self.memory, self.context)
#
#
# cdef class GenotypeInt16Parser(GenotypeParserBase):
#
#     cdef np.int16_t[:, :, :] memory
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
#         self.values = np.empty(shape, dtype='int16')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#     cdef int parse(self) nogil except -1:
#         return genotype_parse(self.memory, self.context)
#
#
# cdef class GenotypeInt32Parser(GenotypeParserBase):
#
#     cdef np.int32_t[:, :, :] memory
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
#         self.values = np.empty(shape, dtype='int32')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#     cdef int parse(self) nogil except -1:
#         return genotype_parse(self.memory, self.context)
#
#
# cdef class GenotypeInt64Parser(GenotypeParserBase):
#
#     cdef np.int64_t[:, :, :] memory
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
#         self.values = np.empty(shape, dtype='int64')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#     cdef int parse(self) nogil except -1:
#         return genotype_parse(self.memory, self.context)
#
#
# cdef inline int genotype_parse(integer[:, :, :] memory,
#                                ParserContext context) nogil except -1:
#     cdef:
#         int allele_index = 0
#
#     # debug('genotype_parse: enter', context)
#
#     # reset temporary buffer
#     context.temp.clear()
#
#     while True:
#
#         if stream.c == SLASH or stream.c == PIPE:
#             genotype_store(memory, context, allele_index)
#             allele_index += 1
#             context.temp.clear()
#
#         elif stream.c == COLON or \
#                 stream.c == TAB or \
#                 stream.c == LF or \
#                 stream.c == CR or \
#                 stream.c == 0:
#             genotype_store(memory, context, allele_index)
#             return 1
#
#         else:
#             temp_append(context)
#
#         stream.getc()
#
#
# cdef inline int genotype_store(integer[:, :, :] memory,
#                                ParserContext context,
#                                int allele_index) nogil except -1:
#     cdef:
#         int success
#
#     # debug('genotype_store: enter', context)
#
#     if allele_index >= context.ploidy:
#         # more alleles than we've made room for, ignore
#         return 0
#
#     # attempt to parse allele
#     success = temp_tolong(context)
#
#     # store value
#     if success:
#         memory[context.chunk_variant_index, context.sample_index, allele_index] = \
#             context.l
#
#     return 1
#
#
# cdef class CalldataParserBase(Parser):
#
#     def __init__(self, ParserContext context, bytes key, fill, number):
#         super(CalldataParserBase, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#         self.number = number
#         self.fill = fill
#
#     def mkchunk(self, chunk, limit=None):
#         field = 'calldata/' + str(<bytes>self.key, 'ascii')
#         values = self.values[:limit]
#         if self.number == 1:
#             values = values.squeeze(axis=2)
#         chunk[field] = values
#         self.malloc()
#
#
# cdef class CalldataInt8Parser(CalldataParserBase):
#
#     cdef np.int8_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='int8')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class CalldataInt16Parser(CalldataParserBase):
#
#     cdef np.int16_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='int16')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class CalldataInt32Parser(CalldataParserBase):
#
#     cdef np.int32_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='int32')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class CalldataInt64Parser(CalldataParserBase):
#
#     cdef np.int64_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_integer_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='int64')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# # TODO unsigned int calldata parsers
#
#
# cdef class CalldataFloat32Parser(CalldataParserBase):
#
#     cdef np.float32_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_floating_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='float32')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class CalldataFloat64Parser(CalldataParserBase):
#
#     cdef np.float64_t[:, :, :] memory
#
#     cdef int parse(self) nogil except -1:
#         return calldata_floating_parse(self.memory, self.number, self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.empty(shape, dtype='float64')
#         self.memory = self.values
#         self.memory[:] = self.fill
#
#
# cdef class CalldataStringParser(Parser):
#
#     cdef np.uint8_t[:] memory
#
#     def __init__(self, ParserContext context, key, dtype, number):
#         super(CalldataStringParser, self).__init__(context)
#         self.key = PyBytes_AS_STRING(key)
#         self.dtype = check_string_dtype(dtype)
#         self.itemsize = self.dtype.itemsize
#         self.number = number
#
#     cdef int parse(self) nogil except -1:
#         cdef:
#             int value_index = 0
#             # index into memory view
#             int memory_offset, memory_index
#             # number of characters read into current value
#             int chars_stored = 0
#
#         # debug('CalldataStringParser.parse: enter', self.context)
#
#         # initialise memory index
#         memory_offset = ((self.context.chunk_variant_index *
#                          self.context.n_samples *
#                          self.number *
#                          self.itemsize) +
#                          (self.context.sample_index *
#                           self.number *
#                           self.itemsize))
#         memory_index = memory_offset
#
#         # read characters until tab
#         while True:
#
#             if self.stream.c == TAB or \
#                     self.stream.c == COLON or \
#                     self.stream.c == CR or \
#                     self.stream.c == LF or \
#                     self.stream.c == 0:
#                 return 1
#
#             elif self.stream.c == COMMA:
#                 # advance value index
#                 value_index += 1
#                 # set memory index to beginning of next item
#                 memory_index = memory_offset + (value_index * self.itemsize)
#                 # reset chars stored
#                 chars_stored = 0
#
#             elif chars_stored < self.itemsize and value_index < self.number:
#                 # store value
#                 self.memory[memory_index] = self.stream.c
#                 # advance memory index
#                 memory_index += 1
#                 # advance number of characters stored
#                 chars_stored += 1
#
#             # advance input stream
#             context_getc(self.context)
#
#     def malloc(self):
#         shape = (self.context.chunk_length, self.context.n_samples, self.number)
#         self.values = np.zeros(shape, dtype=self.dtype)
#         self.memory = self.values.reshape(-1).view('u1')
#
#     def mkchunk(self, chunk, limit=None):
#         field = 'calldata/' + str(<bytes>self.key, 'ascii')
#         values = self.values[:limit]
#         if self.number == 1:
#             values = values.squeeze(axis=2)
#         chunk[field] = values
#         self.malloc()
#

##########################################################################################

#
# cdef class BufferedReader(object):
#
#     cdef:
#         object fileobj
#         int buffer_size
#         bytearray buffer
#         char* stream
#         char* stream_start
#         char* stream_end
#         char c
#
#     def __init__(self, fileobj, buffer_size=2**14):
#         self.fileobj = fileobj
#         self.buffer = bytearray(buffer_size)
#         self.stream_start = PyByteArray_AS_STRING(self.buffer)
#         self.stream = self.stream_start
#         self.fill_buffer()
#         self.getc()
#
#     cdef void fill_buffer(self) nogil:
#         cdef int l
#         with gil:
#             l = self.fileobj.readinto(self.buffer)
#         if l > 0:
#             self.stream = self.stream_start
#             self.stream_end = self.stream + l
#         else:
#             self.stream = NULL
#
#     cdef void getc(self) nogil:
#         if self.stream is self.stream_end:
#             # end of stream buffer
#             self.fill_buffer()
#         if self.stream is NULL:
#             # end of file
#             self.c = 0
#         else:
#             # read next character from stream stream
#             self.c = self.stream[0]
#             self.stream += 1
#
#
# cdef class AsyncBlockParser(object):
#
#     cdef:
#         BufferedReader reader
#         object pool
#         int block_length
#         int block_variant_index
#         int chunk_variant_index
#         char* buffer
#         int buffer_size
#         int buffer_index
#         object async_result
#         int n_lines
#
#     def __cinit__(self, reader, pool, block_length, chunk_variant_index, buffer_size):
#         self.reader = reader
#         self.pool = pool
#         self.block_length = block_length
#         self.block_variant_index = 0
#         self.chunk_variant_index = chunk_variant_index
#         self.buffer_size = buffer_size
#         self.buffer = <char*> malloc(self.buffer_size * sizeof(char*))
#         self.buffer_index = 0
#         self.async_result = None
#         self.n_lines = 0
#
#     def __dealloc__(self):
#         if self.buffer is not NULL:
#             free(self.buffer)
#
#     cdef void reset(self, chunk_variant_index):
#         self.buffer_index = 0
#         self.block_variant_index = 0
#         self.chunk_variant_index = chunk_variant_index
#         self.n_lines = 0
#
#     cdef void grow_buffer(self) nogil:
#         cdef:
#             char* new_buffer
#             cdef int new_buffer_size
#
#         # allocated new buffer
#         new_buffer_size = self.buffer_size * 2
#         new_buffer = <char*> malloc(new_buffer_size * sizeof(char*))
#
#         # copy contents of old buffer to new buffer
#         memcpy(new_buffer, self.buffer, self.buffer_size)
#
#         # free old buffer
#         free(self.buffer)
#         self.buffer = new_buffer
#         self.buffer_size = new_buffer_size
#
#     cdef void append_buffer(self) nogil:
#         if self.buffer_index == self.buffer_size:
#             self.grow_buffer()
#         self.buffer[self.buffer_index] = self.reader.c
#         self.buffer_index += 1
#
#     def wait_completed(self):
#         if self.async_result is not None:
#             self.async_result.get()
#         # otherwise return immediately
#
#     def parse(self):
#
#         # TODO synchronous part - read lines into buffer
#         self.sync_read()
#
#         # TODO async part - parse lines and store data
#         self.async_parse()
#
#     def sync_read(self):
#
#         # TODO universal newlines
#
#         with nogil:
#
#             while True:
#
#                 if self.reader.c == LF:
#                     self.append_buffer()
#                     self.n_lines += 1
#                     if self.n_lines == self.block_length:
#                         # advance beyond end of line for next block
#                         self.reader.getc()
#                         break
#
#                 elif self.reader.c == 0:
#                     self.append_buffer()
#                     self.n_lines += 1
#                     break
#
#                 self.reader.getc()
#
#     def async_parse(self):
#         self.async_result = self.pool.apply_async(block_parse, args=(self,))
#
#
# def block_parse(AsyncBlockParser self):
#     # TODO
#     pass
#
