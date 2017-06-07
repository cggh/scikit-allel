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
        self.variant_n_formats = 0
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
            chrom_parser = VCFChromParser(dtype=types[CHROM_FIELD], store=True)
            fields.remove(CHROM_FIELD)
        else:
            chrom_parser = VCFChromParser(dtype=types[CHROM_FIELD], store=False)
        chrom_parser.malloc_chunk()
        self.chrom_parser = chrom_parser

    def _init_pos(self, fields, types):
        """Setup POS parser."""
        if POS_FIELD in fields:
            if POS_FIELD in types:
                # TODO support user-provided type?
                warn('Only int32 supported for POS field, ignoring requested type: %r'
                     % types[POS_FIELD])
            pos_parser = VCFPosParser(store=True)
            fields.remove(POS_FIELD)
        else:
            pos_parser = VCFPosParser(store=False)
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
            calldata_parser = VCFCallDataParser(formats=format_keys,
                                                types=format_types,
                                                numbers=format_numbers)
        else:
            format_parser = VCFSkipFieldParser()
            calldata_parser = VCFSkipAllCallDataParser()
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
        cdef:
            # index into memory view
            int memory_index = context.chunk_variant_index * self.itemsize
            # number of characters read into current value
            int chars_stored = 0

        # setup context
        context.chrom.clear()
        context.pos = -1
        context.sample_index = 0
        context.sample_field_index = 0
        # check for EOF - important to handle file with no final line terminator
        if stream.c != 0:
            context.variant_index += 1
            context.chunk_variant_index += 1

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

            else:

                # store on context
                context.chrom.append(stream.c)

                # store in chunk
                if self.store and chars_stored < self.itemsize:
                    # store value
                    self.memory[memory_index] = stream.c
                    # advance memory index
                    memory_index += 1
                    # advance number of characters stored
                    chars_stored += 1

            # advance input stream
            stream.getc()

    def malloc_chunk(self, VCFContext context):
        if self.store:
            self.values = np.zeros(context.chunk_length, dtype=self.dtype)
            self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        if self.store:
            chunk[CHROM_FIELD] = self.values[:limit]


cdef class VCFPosParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        np.int32_t[:] memory
        bint store

    def __init__(self, bint store):
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            long value
            int parsed

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
        parsed = vcf_strtol(context.temp, context, &value)

        # store value on context, whatever happens
        context.pos = value

        if parsed > 0 and self.store:
            # store value in chunk
            self.memory[context.chunk_variant_index] = value

    def malloc_chunk(self, VCFContext context):
        if self.store:
            self.values = np.zeros(context.chunk_length, dtype='int32')
            self.memory = self.values
            self.memory[:] = -1

    def make_chunk(self, chunk, limit=None):
        if self.store:
            chunk[POS_FIELD] = self.values[:limit]


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
                stream.getc()
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
            stream.getc()

        return 1

    def malloc_chunk(self, VCFContext context):
        self.values = np.zeros(context.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]


cdef class VCFAltParser(VCFFieldParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, dtype, number):
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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
                stream.getc()
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
            stream.getc()

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    def make_chunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values


cdef class VCFQualParser(VCFFieldParserBase):
    """TODO"""

    cdef np.float32_t[:] memory

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            double value
            int parsed

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
        parsed = vcf_strtol(context.temp, context, &value)

        if parsed > 0:
            # store value
            self.memory[context.chunk_variant_index] = value

    def malloc_chunk(self, VCFContext context):
        self.values = np.empty(context.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = NAN

    def make_chunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]


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
        cdef:
            int filter_index

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return self.parse_missing(stream, context)

        # reset temporary buffer
        context.temp.clear()

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
                stream.getc()
                context.state += 1
                break

            elif stream.c == COMMA or stream.c == COLON or stream.c == SEMICOLON:
                # some of these delimiters are not strictly kosher, but have seen them
                self.parse_filter(context)
                context.temp.clear()

            else:
                context.temp.append(stream.c)

            # advance input stream
            stream.getc()

        return 1

    cdef int parse_missing(self,
                           InputStreamBase stream,
                           VCFContext context) nogil except -1:

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

    cdef int parse_filter(self, VCFContext context) nogil except -1:
        cdef:
            int filter_index
            int i
            char* f

        if context.temp.size == 0:
            warn('empty FILTER', context)
            return 0

        context.temp.terminate()

        # search through filters to find index
        filter_index = cstr_search_sorted(context.temp.data, self.filter_ptrs,
                                          self.n_filters)

        # store value
        if filter_index >= 0:
            self.memory[context.chunk_variant_index, filter_index] = 1

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.n_filters)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    def make_chunk(self, chunk, limit=None):
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]


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

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return self.parse_missing(stream, context)

        # reset buffers
        context.info_key.clear()
        context.info_val.clear()

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
                stream.getc()
                context.state += 1
                break

            elif stream.c == SEMICOLON:
                if context.info_key.size > 0:
                    # handle flag
                    self.parse_info(stream, context)
                else:
                    warn('missing INFO key', context)
                stream.getc()

            elif stream.c == EQUALS:
                # advance input stream beyond '='
                stream.getc()
                if context.info_key.size > 0:
                    self.parse_info(stream, context)
                else:
                    warn('missing INFO key', context)
                    self.skip_parser.parse(stream, context)

            else:

                context.info_key.append(stream.c)
                stream.getc()

    cdef int parse_missing(self,
                           InputStreamBase stream,
                           VCFContext context) nogil except -1:

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
                stream.getc()
                context.state += 1
                break

            # advance input stream
            stream.getc()

    cdef int parse_info(self,
                        InputStreamBase stream,
                        VCFContext context) nogil except -1:

        cdef:
            int parser_index
            PyObject* parser

        # terminate key
        context.info_key.terminate()

        # search for index of current INFO key
        parser_index = cstr_search_sorted(context.info_key.data, self.info_ptrs,
                                          self.n_infos)

        # clear out key for good measure
        context.info_key.clear()

        if parser_index >= 0:
            # obtain parser, use trickery for nogil
            parser = self.info_parser_ptrs[parser_index]
            (<VCFInfoParserBase> parser).parse(stream, context)
        else:
            self.skip_parser.parse(stream, context)

    def malloc_chunk(self, VCFContext context):
        for parser in self.info_parsers:
            parser.malloc_chunk()

    def make_chunk(self, chunk, limit=None):
        for parser in self.info_parsers:
            parser.make_chunk(chunk, limit=limit)


cdef class VCFInfoParserBase(object):
    """TODO"""

    cdef:
        bytes key
        object dtype
        int itemsize
        int number
        object fill
        object values

    def __init__(self, key=None, dtype=None, number=1, fill=0):
        self.key = key
        if dtype is not None:
            dtype = np.dtype(dtype)
            self.itemsize = dtype.itemsize
        else:
            self.dtype = dtype
            self.itemsize = 0
        self.number = number
        self.fill = fill

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    def make_chunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values

    cdef int parse_integer(self,
                           InputStreamBase stream,
                           VCFContext context,
                           integer[:, :] memory) nogil except -1:
        cdef:
            int value_index = 0

        # reset temporary buffer
        context.info_val.clear()

        while True:

            if stream.c == 0 or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == TAB or \
                    stream.c == SEMICOLON:
                self.store_integer(self, context, value_index, memory)
                break

            elif stream.c == COMMA:
                self.store_integer(self, context, value_index, memory)
                context.info_val.clear()
                value_index += 1

            else:
                context.info_val.append(stream.c)

            stream.getc()

    cdef int store_integer(self,
                           VCFContext context,
                           int value_index,
                           integer[:, :] memory) nogil except -1:
        cdef:
            int parsed
            long value

        if value_index >= self.number:
            # more values than we have room for, ignore
            return 0

        # parse string as integer
        parsed = vcf_strtol(context.info_val, context, &value)

        if parsed > 0:
            # store value
            memory[context.chunk_variant_index, value_index] = value

    cdef int parse_floating(self,
                            InputStreamBase stream,
                            VCFContext context,
                            floating[:, :] memory) nogil except -1:
        cdef:
            int value_index = 0

        # reset temporary buffer
        context.info_val.clear()

        while True:

            if stream.c == 0 or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == TAB or \
                    stream.c == SEMICOLON:
                self.store_floating(self, context, value_index, memory)
                break

            elif stream.c == COMMA:
                self.store_floating(self, context, value_index, memory)
                context.info_val.clear()
                value_index += 1

            else:
                context.info_val.append(stream.c)

            stream.getc()

    cdef int store_floating(self,
                           VCFContext context,
                           int value_index,
                           floating[:, :] memory) nogil except -1:
        cdef:
            int parsed
            double value

        if value_index >= self.number:
            # more values than we have room for, ignore
            return 0

        # parse string as floating
        parsed = vcf_strtod(context.info_val, context, &value)

        if parsed > 0:
            # store value
            memory[context.chunk_variant_index, value_index] = value


cdef class VCFInfoInt32Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.int32_t[:, :] memory

    def __init__(self, key, number=1, fill=-1):
        super(VCFInfoInt32Parser, self).__init__(key, dtype='int32', number=number,
                                                 fill=fill)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return self.parse_integer(stream, context, self.memory)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoInt64Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.int64_t[:, :] memory

    def __init__(self, key, number=1, fill=-1):
        super(VCFInfoInt64Parser, self).__init__(key, dtype='int64', number=number,
                                                 fill=fill)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return self.parse_integer(stream, context, self.memory)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat32Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.float32_t[:, :] memory

    def __init__(self, key, number=1, fill=NAN):
        super(VCFInfoFloat32Parser, self).__init__(key, dtype='float32', number=number,
                                                   fill=fill)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return self.parse_floating(stream, context, self.memory)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat64Parser(VCFInfoParserBase):
    """TODO"""

    cdef np.float64_t[:, :] memory

    def __init__(self, key, number=1, fill=NAN):
        super(VCFInfoFloat64Parser, self).__init__(key, dtype='float64', number=number,
                                                   fill=fill)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return self.parse_floating(stream, context, self.memory)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFlagParser(VCFInfoParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, key):
        super(VCFInfoFlagParser, self).__init__(key, dtype='uint8')

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        # nothing to parse
        self.memory[context.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while stream.c != SEMICOLON and \
                stream.c != TAB and \
                stream.c != LF and \
                stream.c != CR and \
                stream.c != 0:
            stream.getc()

    def make_chunk(self, chunk, limit=None):
        # override to view as bool array
        field = 'variants/' + str(self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)

    def malloc_chunk(self, VCFContext context):
        self.values = np.empty(context.chunk_length, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoStringParser(VCFInfoParserBase):
    """TODO"""

    cdef np.uint8_t[:] memory

    def __init__(self, key, dtype, number=1):
        dtype = check_string_dtype(dtype)
        super(VCFInfoStringParser, self).__init__(key, dtype=dtype, number=number)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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
            stream.getc()

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')


cdef class VCFInfoSkipParser(VCFInfoParserBase):
    """TODO"""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        # ensure we advance the end of the field
        while stream.c != SEMICOLON and \
                stream.c != TAB and \
                stream.c != LF and \
                stream.c != CR and \
                stream.c != 0:
            stream.getc()

    def malloc_chunk(self, VCFContext context):
        pass

    def make_chunk(self, chunk, limit=None):
        pass


cdef class VCFFormatParser(VCFFieldParserBase):
    """TODO"""

    cdef:
        tuple formats
        int n_formats
        char** format_ptrs

    def __cinit__(self, formats):

        # setup FORMAT keys
        self.formats = tuple(sorted(formats))
        self.n_formats = len(self.formats)

        # setup FORMAT keys as C strings for nogil searching
        self.format_ptrs = <char**> malloc(sizeof(char*) * self.n_formats)
        for i in range(self.n_formats):
            self.format_ptrs[i] = <char*> self.formats[i]

    def __dealloc__(self):
        if self.format_ptrs is not NULL:
            free(self.format_ptrs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int i

        # reset temporary buffer
        context.temp.clear()
        context.variant_format_indices.clear()

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
                stream.getc()
                break

            elif stream.c == COLON:
                self.store_format(context)

            else:
                context.temp.append(stream.c)

            # advance to next character
            stream.getc()

    cdef int store_format(self, VCFContext context) nogil except -1:
        cdef int format_index

        # deal with empty or missing data
        if context.temp.size == 0:
            warn('empty FORMAT', context)
            return 0

        if context.temp.size == 1 and context.temp[0] == PERIOD:
            return 0

        # terminate the string
        context.temp.terminate()

        # find format index
        format_index = cstr_search_sorted(context.temp.data, self.format_ptrs,
                                          self.n_formats)

        # add to vector of indices for the current variant
        context.variant_format_indices.append(format_index)



cdef class VCFSkipAllCallDataParser(VCFFieldParserBase):
    """Skip a field."""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            # advance input stream
            stream.getc()


cdef class VCFCallDataParser(object):
    """TODO"""

    cdef:
        tuple formats
        int n_formats
        tuple parsers
        PyObject** parser_ptrs
        VCFCallDataParserBase skip_parser

    def __cinit__(self, formats, types, numbers):

        # setup formats
        self.formats = tuple(sorted(formats))
        self.n_formats = len(self.formats)

        # setup parsers
        self.skip_parser = VCFCallDataSkipParser()
        parsers = list()
        for key in self.formats:
            t = types[key]
            n = numbers[key]

            # special handling of "genotype" dtypes for any field
            if isinstance(t, str) and t.startswith('genotype/'):
                t = np.dtype(t.split('/')[1])
                if t == np.dtype('int8'):
                    parser = VCFGenotypeInt8Parser(key, fill=-1)
                elif t == np.dtype('int16'):
                    parser = VCFGenotypeInt16Parser(key, fill=-1)
                elif t == np.dtype('int32'):
                    parser = VCFGenotypeInt32Parser(key, fill=-1)
                elif t == np.dtype('int64'):
                    parser = VCFGenotypeInt64Parser(key, fill=-1)
                else:
                    warnings.warn('type %r not supported for genotype field %r, '
                                  'field will be skipped' % (t, key))
                    parser = self.skip_parser

            # special handling of GT field
            elif key == b'GT' and t == np.dtype('int8'):
                parser = VCFGenotypeInt8Parser(key, fill=-1)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = VCFGenotypeInt16Parser(key, fill=-1)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = VCFGenotypeInt32Parser(key, fill=-1)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = VCFGenotypeInt64Parser(key, fill=-1)

            # all other calldata
            elif t == np.dtype('int8'):
                parser = VCFCallDataInt8Parser(key, number=n, fill=-1)
            elif t == np.dtype('int16'):
                parser = VCFCallDataInt16Parser(key, number=n, fill=-1)
            elif t == np.dtype('int32'):
                parser = VCFCallDataInt32Parser(key, number=n, fill=-1)
            elif t == np.dtype('int64'):
                parser = VCFCallDataInt64Parser(key, number=n, fill=-1)
            elif t == np.dtype('float32'):
                parser = VCFCallDataFloat32Parser(key, number=n, fill=NAN)
            elif t == np.dtype('float64'):
                parser = VCFCallDataFloat64Parser(key, number=n, fill=NAN)
            elif t.kind == 'S':
                parser = VCFCallDataStringParser(key, dtype=t, number=n)

            # TODO unsigned int parsers

            else:
                parser = self.skip_parser
                warnings.warn('type %r not supported for FORMAT field %r, field will be '
                              'skipped' % (t, key))

            parsers.append(parser)
        self.parsers = tuple(parsers)

        # store pointers to parsers
        self.parser_ptrs = <PyObject**> malloc(sizeof(PyObject*) * self.n_formats)
        for i in range(self.n_formats):
            self.parser_ptrs[i] = <PyObject*> self.parsers[i]

    def __dealloc__(self):
        if self.parser_ptrs is not NULL:
            free(self.parser_ptrs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int i
            PyObject* parser

        # initialise context
        context.sample_index = 0
        context.sample_field_index = 0

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                return 1

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                return 1

            elif stream.c == TAB:
                context.sample_index += 1
                context.sample_field_index = 0
                stream.getc()

            elif stream.c == COLON:
                context.sample_field_index += 1
                stream.getc()

            elif context.sample_field_index < context.variant_format_indices.size:
                format_index = context.variant_format_indices.data[context.sample_field_index]

                if format_index >= 0:
                    parser = self.parser_ptrs[format_index]
                    # jump through some hoops to avoid references (which need the GIL)
                    (<VCFCallDataParserBase>parser).parse(stream, context)

                else:
                    self.skip_parser.parse(stream, context)

            else:
                # more fields than formats declared for this variant
                self.skip_parser.parse(stream, context)

    def malloc_chunk(self, VCFContext context):
        for parser in self.parsers:
            parser.malloc_chunk(context)

    def make_chunk(self, chunk, limit=None):
        for parser in self.parsers:
            parser.make_chunk(chunk, limit=limit)


cdef class VCFCallDataParserBase(object):

    cdef:
        bytes key
        object dtype
        int itemsize
        int number
        object fill
        object values

    def __init__(self, key=None, dtype=None, number=1, fill=0):
        self.key = key
        if dtype is not None:
            dtype = np.dtype(dtype)
            self.itemsize = dtype.itemsize
        else:
            self.dtype = dtype
            self.itemsize = 0
        self.number = number
        self.fill = fill

    def make_chunk(self, chunk, limit=None):
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    cdef int parse_integer(self,
                           InputStreamBase stream,
                           VCFContext context,
                           integer[:, :, :] memory) nogil except -1:

        cdef:
            int value_index = 0

        # reset temporary buffer
        context.temp.clear()

        while True:

            if stream.c == COMMA:
                self.store_integer(context, value_index, memory)
                context.temp.clear()
                value_index += 1

            elif stream.c == COLON or \
                    stream.c == TAB or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == 0:
                self.store_integer(context, value_index, memory)
                break

            else:
                context.temp.append(stream.c)

            stream.getc()

    cdef int store_integer(self,
                           VCFContext context,
                           int value_index,
                           integer[:, :, :] memory) nogil except -1:
        cdef:
            int parsed
            long value

        if value_index >= self.number:
            # more values than we have room for, ignore
            return 0

        parsed = vcf_strtol(context.temp, context, &value)

        # store value
        if parsed > 0:
            memory[context.chunk_variant_index, context.sample_index, value_index] = value

    cdef int parse_floating(self,
                            InputStreamBase stream,
                            VCFContext context,
                            floating[:, :, :] memory) nogil except -1:

        cdef:
            int value_index = 0

        # reset temporary buffer
        context.temp.clear()

        while True:

            if stream.c == COMMA:
                self.store_floating(context, value_index, memory)
                context.temp.clear()
                value_index += 1

            elif stream.c == COLON or \
                    stream.c == TAB or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == 0:
                self.store_floating(context, value_index, memory)
                break

            else:
                context.temp.append(stream.c)

            stream.getc()

    cdef int store_floating(self,
                            VCFContext context,
                            int value_index,
                            floating[:, :, :] memory) nogil except -1:
        cdef:
            int parsed
            double value

        if value_index >= self.number:
            # more values than we have room for, ignore
            return 0

        parsed = vcf_strtod(context.temp, context, &value)

        # store value
        if parsed > 0:
            memory[context.chunk_variant_index, context.sample_index, value_index] = value


cdef class VCFCallDataSkipParser(VCFCallDataParserBase):

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        while stream.c != COLON and \
                stream.c != TAB and \
                stream.c != CR and \
                stream.c != LF and \
                stream.c != 0:
            stream.getc()


cdef class VCFGenotypeInt8Parser(VCFCallDataParserBase):

    cdef:
        np.int8_t[:, :, :] memory

    def __init__(self, key, dtype, ploidy=2):
        super(VCFGenotypeInt8Parser, self).__init__(key, dtype='int8', number=ploidy,
                                                    fill=-1)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory, self.number)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, context.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt16Parser(VCFCallDataParserBase):

    cdef:
        np.int16_t[:, :, :] memory

    def __init__(self, key, dtype, ploidy=2):
        super(VCFGenotypeInt16Parser, self).__init__(key, dtype='int16', number=ploidy,
                                                     fill=-1)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory, self.number)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, context.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt32Parser(VCFCallDataParserBase):

    cdef:
        np.int32_t[:, :, :] memory

    def __init__(self, key, dtype, ploidy=2):
        super(VCFGenotypeInt32Parser, self).__init__(key, dtype='int32', number=ploidy,
                                                     fill=-1)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory, self.number)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, context.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt64Parser(VCFCallDataParserBase):

    cdef:
        np.int64_t[:, :, :] memory

    def __init__(self, key, dtype, ploidy=2):
        super(VCFGenotypeInt64Parser, self).__init__(key, dtype='int64', number=ploidy,
                                                     fill=-1)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory, self.number)

    def malloc_chunk(self, VCFContext context):
        shape = (context.chunk_length, context.n_samples, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_genotype_parse(InputStreamBase stream,
                            VCFContext context,
                            integer[:, :, :] memory,
                            int ploidy) nogil except -1:
    cdef:
        int allele_index = 0

    # reset temporary buffer
    context.temp.clear()

    while True:

        if stream.c == SLASH or stream.c == PIPE:
            vcf_genotype_store(context, memory, allele_index, ploidy)
            allele_index += 1
            context.temp.clear()

        elif stream.c == COLON or \
                stream.c == TAB or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == 0:
            vcf_genotype_store(context, memory, allele_index, ploidy)
            break

        else:
            context.temp.append(stream.c)

        stream.getc()


cdef int vcf_genotype_store(VCFContext context,
                            integer[:, :, :] memory,
                            int allele_index,
                            int ploidy) nogil except -1:
    cdef:
        int parsed
        long allele

    if allele_index >= ploidy:
        # more alleles than we've made room for, ignore
        return 0

    # attempt to parse allele
    parsed = vcf_strtol(context.temp, context, &allele)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_index, allele_index] = allele


##########################################################################################
# Low-level VCF value parsing functions


cdef int vcf_strtol(CharVector value, VCFContext context, long* l) nogil:
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
    value.terminate()

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


cdef int vcf_strtod(CharVector value, VCFContext context, double* d) nogil:
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
    value.terminate()

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



# cdef class CallDataParserBase(Parser):
#
#     def __init__(self, ParserContext context, bytes key, fill, number):
#         super(CallDataParserBase, self).__init__(context)
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
# cdef class CallDataInt8Parser(CallDataParserBase):
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
# cdef class CallDataInt16Parser(CallDataParserBase):
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
# cdef class CallDataInt32Parser(CallDataParserBase):
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
# cdef class CallDataInt64Parser(CallDataParserBase):
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
# cdef class CallDataFloat32Parser(CallDataParserBase):
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
# cdef class CallDataFloat64Parser(CallDataParserBase):
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
# cdef class CallDataStringParser(Parser):
#
#     cdef np.uint8_t[:] memory
#
#     def __init__(self, ParserContext context, key, dtype, number):
#         super(CallDataStringParser, self).__init__(context)
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
#         # debug('CallDataStringParser.parse: enter', self.context)
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
#             stream.getc()
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
