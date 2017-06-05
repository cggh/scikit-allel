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
# CONSTANTS


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
# FUSED TYPES


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
# GENERAL I/O


cdef class CharBuffer(object):
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
        """Double the capacity if the buffer is full."""
        if self.size >= self.capacity:
            self.capacity *= 2
            self.data = <char*> realloc(self.data, sizeof(char) * self.capacity)

    cdef void append(self, char c) nogil:
        """Append a single char to the buffer."""
        self.grow_if_full()
        self.data[self.size] = c
        self.size += 1

    cdef void terminate(self) nogil:
        """Terminate the buffer by appending a null byte."""
        self.append(0)

    cdef void clear(self) nogil:
        """Cheaply clear the buffer by setting the size to 0."""
        self.size = 0

    cdef bytes to_pybytes(self):
        return PyBytes_FromStringAndSize(self.data, self.size)


cdef class InputStream(object):
    """Abstract base class defining an input stream over C chars."""

    cdef:
        # character at the current position in the stream
        char c

    cdef int getc(self) nogil except -1:
        """Read the next character from the stream and store it in the `c` attribute."""
        pass


cdef class FileInputStream(InputStream):

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

    cdef int read_line_into(self, CharBuffer dest) nogil except -1:
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

    cdef int read_lines_into(self, CharBuffer dest, int n) nogil except -1:
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
        cdef CharBuffer line = CharBuffer()
        self.read_line_into(line)
        return line.to_pybytes()


cdef class CharBufferInputStream(InputStream):

    cdef:
        CharBuffer buffer
        int stream_index
        int buffer_size

    def __cinit__(self, CharBuffer buffer):
        self.buffer = buffer
        self.stream_index = 0
        self.getc()

    cdef int getc(self) nogil except -1:
        if self.stream_index < self.buffer.size:
            self.c = self.buffer.data[self.stream_index]
            self.stream_index += 1
        else:
            self.c = 0


##########################################################################################
# VCF PARSING UTILITIES


cdef class VCFContext(object):

    cdef:

        # temporary buffer
        CharBuffer temp
        # buffers used for INFO parsing
        CharBuffer info_key
        CharBuffer info_val
        # overall parser state
        int state
        # static attributes - should not change during parsing
        int chunk_length
        int n_samples
        int ploidy
        # dynamic attributes - reflect current position during parsing
        int variant_index  # index of current variant
        int chunk_variant_index  # index of current variant within chunk
        int sample_index  # index of current sample within calldata
        int sample_field_index  # index of field within calldata for current sample
        int** variant_format_indices  # indices of formats for the current variant

    def __cinit__(self, int n_samples, int chunk_length, int ploidy):
        # TODO
        pass

    def __dealloc__(self):
        # TODO
        pass


cdef class VCFChunkBuilder(object):

    def __init__(self, *args, **kwargs):
        # TODO
        pass

    def parse(self, InputStream stream, VCFContext context, int n_lines=None):
        """Parse up to `n_lines` or end of current chunk, whichever comes sooner."""
        # TODO
        pass

    def get_chunk(self):
        # TODO make chunk
        pass


# TODO ABC for Parsers


cdef long vcf_buffer_tolong(CharBuffer buffer, long default, VCFContext context) nogil:
    # TODO
    pass

    # cdef bytes to_pybytes(self):
    #     """Convert the buffer to a Python bytes object."""
    #     return PyBytes_FromStringAndSize(self.data, self.size)
    #
    # cdef long to_long(self, long default):
    #     """Parse the buffer as a long value."""
    #     cdef:
    #         char* str_end
    #         int chars_parsed






##########################################################################################
# LOGGING


cdef int warn(message, ParserContext context) nogil:
    with gil:
        # TODO customize message based on state (CHROM, POS, etc.)
        message += '; variant index: %s' % context.variant_index
        b = PyBytes_FromStringAndSize(context.temp, context.temp_size)
        message += '; temporary buffer: %s' % b
        warnings.warn(message)


cdef int debug(msg, ParserContext context=None) nogil except -1:
    with gil:
        msg = '[DEBUG] ' + str(msg) + '\n'
        if context is not None:
            msg += 'state: %s' % context.state
            msg += '; variant_index: %s' % context.variant_index
            msg += '; chunk_variant_index: %s' % context.chunk_variant_index
            msg += '; sample_index: %s' % context.sample_index
            msg += '; format_index: %s' % context.format_index
            b = PyBytes_FromStringAndSize(context.temp, context.temp_size)
            msg += '; temp: %s' % b
            msg += '; c: %s' % <bytes>context.c
            msg += '; n_formats: %s' % context.n_formats
            msg += '; variant_n_formats: %s' % context.variant_n_formats
            msg += '; calldata_parsers: %s' % context.calldata_parsers
        print(msg, file=sys.stderr)
        sys.stderr.flush()


##########################################################################################
# LEGACY


cdef class VCFChunkIterator(object):

    cdef:
        ParserContext context
        Parser chrom_parser
        Parser pos_parser
        Parser id_parser
        Parser ref_parser
        Parser alt_parser
        Parser qual_parser
        Parser filter_parser
        Parser info_parser
        Parser format_parser
        Parser calldata_parser

    def __init__(self,
                 input_file,
                 int input_buffer_size,
                 int chunk_length,
                 int temp_buffer_size,
                 headers,
                 fields,
                 types,
                 numbers,
                 ploidy=2):
        # debug('VCFChunkIterator.__init__: enter')

        # setup context
        n_samples = len(headers.samples)
        context = ParserContext(input_file=input_file,
                                input_buffer_size=input_buffer_size,
                                temp_buffer_size=temp_buffer_size,
                                n_samples=n_samples,
                                chunk_length=chunk_length,
                                ploidy=ploidy)
        self.context = context

        # copy so we don't modify someone else's data
        fields = set(fields)

        # setup CHROM parser
        if CHROM_FIELD in fields:
            chrom_parser = ChromParser(context, dtype=types[CHROM_FIELD], skip=False)
            fields.remove(CHROM_FIELD)
        else:
            chrom_parser = ChromParser(context, dtype=None, skip=True)
        chrom_parser.malloc()
        self.chrom_parser = chrom_parser

        # setup POS parser
        if POS_FIELD in fields:
            # TODO user-provided type
            pos_parser = PosInt32Parser(context)
            fields.remove(POS_FIELD)
        else:
            pos_parser = SkipPosParser(context)
        pos_parser.malloc()
        self.pos_parser = pos_parser

        # setup ID parser
        if ID_FIELD in fields:
            id_parser = StringFieldParser(context, field=ID_FIELD, dtype=types[ID_FIELD])
            fields.remove(ID_FIELD)
        else:
            id_parser = SkipFieldParser(context)
        id_parser.malloc()
        self.id_parser = id_parser

        # setup REF parser
        if REF_FIELD in fields:
            ref_parser = StringFieldParser(context, field=REF_FIELD, dtype=types[REF_FIELD])
            fields.remove(REF_FIELD)
        else:
            ref_parser = SkipFieldParser(context)
        ref_parser.malloc()
        self.ref_parser = ref_parser

        # setup ALT parser
        if ALT_FIELD in fields:
            t = types[ALT_FIELD]
            n = numbers[ALT_FIELD]
            alt_parser = AltParser(context, dtype=t, number=n)
            fields.remove(ALT_FIELD)
        else:
            alt_parser = SkipFieldParser(context)
        alt_parser.malloc()
        self.alt_parser = alt_parser

        # setup QUAL parser
        if QUAL_FIELD in fields:
            # TODO user-provided type
            qual_parser = QualFloat32Parser(context, fill=-1)
            fields.remove(QUAL_FIELD)
        else:
            qual_parser = SkipFieldParser(context)
        qual_parser.malloc()
        self.qual_parser = qual_parser

        # setup FILTER parser
        filter_keys = list()
        for field in list(fields):
            if field.startswith('variants/FILTER_'):
                filter = field[16:].encode('ascii')
                filter_keys.append(filter)
                fields.remove(field)
        # debug(filter_keys, context)
        if filter_keys:
            filter_parser = FilterParser(context, filters=filter_keys)
        else:
            filter_parser = SkipFieldParser(context)
        filter_parser.malloc()
        self.filter_parser = filter_parser

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
            info_parser = InfoParser(context, infos=info_keys, types=info_types,
                                     numbers=info_numbers)
        else:
            info_parser = SkipFieldParser(context)
        info_parser.malloc()
        self.info_parser = info_parser

        # setup FORMAT and calldata parsers
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
        # debug('iter_vcf format_keys: %s' % str(format_keys), context)
        if format_keys:
            format_parser = FormatParser(context)
            calldata_parser = CalldataParser(context,
                                             formats=format_keys,
                                             types=format_types,
                                             numbers=format_numbers)
        else:
            format_parser = SkipFieldParser(context)
            calldata_parser = SkipAllCalldataParser(context)
        format_parser.malloc()
        calldata_parser.malloc()
        self.format_parser = format_parser
        self.calldata_parser = calldata_parser

        if fields:
            # shouldn't ever be any left over
            raise RuntimeError('unexpected fields left over: %r' % set(fields))

        # debug('VCFChunkIterator.__init__: exit')

    def __iter__(self):
        return self

    def __next__(self):
        return next_chunk(self, self.context)


cpdef next_chunk(VCFChunkIterator self, ParserContext context):

    # debug('VCFChunkIterator.__next__: enter', context)

    with nogil:

        while True:

            if context.state == ParserState.EOF:
                break

            elif context.state == ParserState.EOL:

                # debug('VCFChunkIterator.__next__: EOL', context)

                # decide whether to start a new chunk
                if context.chunk_variant_index + 1 == context.chunk_length:
                    # need to return a chunk
                    break

                # debug('handle line terminators')
                if context.c == LF:
                    context_getc(context)
                elif context.c == CR:
                    context_getc(context)
                    if context.c == LF:
                        context_getc(context)
                else:
                    # shouldn't ever happen
                    warn('unexpected EOL character', context)
                    break

                # debug('advance state')
                context.state = ParserState.CHROM

            elif context.state == ParserState.CHROM:
                # debug('VCFChunkIterator.__next__: CHROM', context)
                self.chrom_parser.parse()

            elif context.state == ParserState.POS:
                # debug('VCFChunkIterator.__next__: POS', context)
                self.pos_parser.parse()

            elif context.state == ParserState.ID:
                # debug('VCFChunkIterator.__next__: ID', context)
                self.id_parser.parse()

            elif context.state == ParserState.REF:
                # debug('VCFChunkIterator.__next__: REF', context)
                self.ref_parser.parse()

            elif context.state == ParserState.ALT:
                # debug('VCFChunkIterator.__next__: ALT', context)
                self.alt_parser.parse()

            elif context.state == ParserState.QUAL:
                # debug('VCFChunkIterator.__next__: QUAL', context)
                self.qual_parser.parse()

            elif context.state == ParserState.FILTER:
                # debug('VCFChunkIterator.__next__: FILTER', context)
                self.filter_parser.parse()

            elif context.state == ParserState.INFO:
                # debug('VCFChunkIterator.__next__: INFO', context)
                self.info_parser.parse()

            elif context.state == ParserState.FORMAT:
                # debug('VCFChunkIterator.__next__: FORMAT', context)
                self.format_parser.parse()

            elif context.state == ParserState.CALLDATA:
                # debug('VCFChunkIterator.__next__: CALLDATA', context)
                self.calldata_parser.parse()

            else:
                # shouldn't ever happen
                warn('unexpected parser state', context)
                break

    chunk_length = context.chunk_variant_index + 1
    if chunk_length > 0:
        if chunk_length < context.chunk_length:
            limit = chunk_length
        else:
            limit = None
        chunk = dict()
        self.chrom_parser.mkchunk(chunk, limit=limit)
        self.pos_parser.mkchunk(chunk, limit=limit)
        self.id_parser.mkchunk(chunk, limit=limit)
        self.ref_parser.mkchunk(chunk, limit=limit)
        self.alt_parser.mkchunk(chunk, limit=limit)
        self.qual_parser.mkchunk(chunk, limit=limit)
        self.filter_parser.mkchunk(chunk, limit=limit)
        self.info_parser.mkchunk(chunk, limit=limit)
        self.calldata_parser.mkchunk(chunk, limit=limit)
        self.context.chunk_variant_index = -1
        return chunk

    else:
        raise StopIteration


cdef enum ParserState:
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


cdef int MAX_FORMATS = 100


cdef class ParserContext:
    cdef:
        # input file and buffer
        object input_file
        int input_buffer_size
        bytearray input_buffer
        char* input
        char* input_start
        char* input_end
        # temporary buffer
        int temp_buffer_size
        bytearray temp_buffer
        char* temp
        int temp_size
        # state
        int state
        char c
        long l
        double d
        int n_samples
        int variant_index
        int chunk_variant_index
        int sample_index
        int chunk_length
        int ploidy
        # infos
        int n_infos
        tuple infos
        char** info_ptrs
        list info_parsers
        PyObject** info_parser_ptrs
        # filters
        int n_filters
        tuple filters
        char** filter_ptrs
        # formats
        int n_formats
        tuple formats
        char** format_ptrs
        int variant_n_formats
        int format_index
        list calldata_parsers
        PyObject** calldata_parser_ptrs
        PyObject** variant_calldata_parser_ptrs


    def __cinit__(self,
                  input_file,
                  int input_buffer_size,
                  int temp_buffer_size,
                  int n_samples,
                  int chunk_length,
                  int ploidy):

        # initialize input buffer
        self.input_file = input_file
        self.input_buffer_size = input_buffer_size
        self.input_buffer = bytearray(input_buffer_size)
        self.input_start = PyByteArray_AS_STRING(self.input_buffer)
        self.input = self.input_start
        context_fill_buffer(self)
        context_getc(self)

        # initialize temporary buffer
        self.temp_buffer = bytearray(temp_buffer_size)
        self.temp = PyByteArray_AS_STRING(self.temp_buffer)
        self.temp_size = 0

        # initialize pointer arrays
        self.variant_calldata_parser_ptrs = <PyObject**> malloc(MAX_FORMATS *
                                                                sizeof(PyObject*))

        # initialize state
        self.state = ParserState.CHROM
        self.n_samples = n_samples
        self.variant_index = -1
        self.chunk_variant_index = -1
        self.sample_index = 0
        self.format_index = 0
        self.chunk_length = chunk_length
        self.ploidy = ploidy

    def __dealloc__(self):

        # infos
        if self.info_ptrs is not NULL:
            free(self.info_ptrs)
        if self.info_parser_ptrs is not NULL:
            free(self.info_parser_ptrs)

        # filters
        if self.filter_ptrs is NULL:
            free(self.filter_ptrs)

        # calldata
        if self.format_ptrs is not NULL:
            free(self.format_ptrs)
        if self.calldata_parser_ptrs is not NULL:
            free(self.calldata_parser_ptrs)
        if self.variant_calldata_parser_ptrs is not NULL:
            free(self.variant_calldata_parser_ptrs)


cdef inline int context_fill_buffer(ParserContext context) nogil:
    cdef:
        int l
    with gil:
        l = context.input_file.readinto(context.input_buffer)
    if l > 0:
        context.input = context.input_start
        context.input_end = context.input + l
        return 1
    else:
        context.input = NULL
        return 0


cdef inline int context_getc(ParserContext context) nogil:

    if context.input is context.input_end:
        # end of input buffer
        context_fill_buffer(context)

    if context.input is NULL:
        # end of file
        context.c = 0
        return 0

    else:
        # read next character from input buffer and advance buffer
        context.c = context.input[0]
        context.input += 1
        return 1


cdef inline void temp_clear(ParserContext context) nogil:
    context.temp_size = 0


cdef inline int temp_append(ParserContext context) nogil:

    # if context.temp_size >= context.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # store current character
    context.temp[context.temp_size] = context.c

    # increase size
    context.temp_size += 1

    return 1


cdef inline int temp_terminate(ParserContext context) nogil:

    # if context.temp_size >= context.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    context.temp[context.temp_size] = 0

    return 1


cdef inline int temp_tolong(ParserContext context) nogil:
    cdef:
        char* str_end
        int parsed

    if context.temp_size == 0:

        warn('expected integer, found empty value', context)
        return 0

    if context.temp_size == 1 and context.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # terminate string
    temp_terminate(context)

    # do parsing
    context.l = strtol(context.temp, &str_end, 10)

    # check success
    parsed = str_end - context.temp

    # check success
    if context.temp_size == parsed:

        return 1

    else:

        if parsed > 0:
            warn('not all characters parsed for integer value', context)
            return 1

        else:
            warn('error parsing integer value', context)
            return 0


cdef inline int temp_todouble(ParserContext context) nogil:
    cdef:
        char* str_end
        int parsed

    if context.temp_size == 0:

        warn('expected floating point number, found empty value', context)
        return 0

    if context.temp_size == 1 and context.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # terminate string
    temp_terminate(context)

    # do parsing
    context.d = strtod(context.temp, &str_end)

    # check success
    parsed = str_end - context.temp

    # check success
    if context.temp_size == parsed:

        return 1

    else:

        if parsed > 0:
            warn('not all characters parsed for floating value', context)
            return 1

        else:
            warn('error parsing floating value', context)
            return 0


cdef class Parser(object):
    """Abstract base class."""

    cdef ParserContext context
    cdef char* key
    cdef int number
    cdef object values
    cdef object fill
    cdef object dtype
    cdef int itemsize

    def __init__(self, ParserContext context):
        # debug('Parser.__init__: enter', context)
        self.context = context

    cdef int parse(self) nogil except -1:
        # debug('Parser.parse: enter', self.context)
        pass

    def malloc(self):
        # debug('Parser.malloc: enter', self.context)
        pass

    def mkchunk(self, chunk, limit=None):
        # debug('Parser.mkchunk: enter', self.context)
        pass


def check_string_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != 'S':
        raise ValueError('expected byte string ("S") dtype, found: %r' % dtype)
    return dtype


cdef class ChromParser(Parser):

    cdef np.uint8_t[:] memory
    cdef bint skip

    def __init__(self, ParserContext context, dtype, skip):
        super(ChromParser, self).__init__(context)
        if not skip:
            self.dtype = check_string_dtype(dtype)
            self.itemsize = self.dtype.itemsize
        self.skip = skip

    cdef int parse(self) nogil except -1:
        return chrom_parse(self.memory, self.itemsize, self.context, self.skip)

    def malloc(self):
        if not self.skip:
            self.values = np.zeros(self.context.chunk_length, dtype=self.dtype)
            self.memory = self.values.view('u1')

    def mkchunk(self, chunk, limit=None):
        if not self.skip:
            chunk[CHROM_FIELD] = self.values[:limit]
            self.malloc()


cdef inline int chrom_parse(np.uint8_t[:] memory,
                            int itemsize,
                            ParserContext context,
                            bint skip) nogil except -1:
    cdef:
        # index into memory view
        int memory_index
        # number of characters read into current value
        int chars_stored = 0

    # TODO store CHROM on context

    # check for EOF
    if context.c != 0:
        context.variant_index += 1
        context.chunk_variant_index += 1

    # initialise memory index
    memory_index = context.chunk_variant_index * itemsize

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            return 0

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            return 0

        elif context.c == TAB:
            # advance input stream beyond tab
            context_getc(context)
            # advance to next field
            context.state += 1
            return 1

        else:

            if not skip and chars_stored < itemsize:
                # store value
                memory[memory_index] = context.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

        # advance input stream
        context_getc(context)


cdef class StringFieldParser(Parser):
    """Generic string field parser, used for CHROM, ID, REF."""

    cdef np.uint8_t[:] memory
    cdef object field

    def __init__(self, ParserContext context, field, dtype):
        super(StringFieldParser, self).__init__(context)
        self.field = field
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize

    cdef int parse(self) nogil except -1:
        return string_field_parse(self.memory, self.itemsize, self.context)

    def malloc(self):
        self.values = np.zeros(self.context.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    def mkchunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]
        self.malloc()


cdef inline int string_field_parse(np.uint8_t[:] memory,
                                   int itemsize,
                                   ParserContext context) nogil except -1:
    cdef:
        # index into memory view
        int memory_index
        # number of characters read into current value
        int chars_stored = 0

    # initialise memory index
    memory_index = context.chunk_variant_index * itemsize

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            break

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            break

        elif context.c == TAB:
            # advance input stream beyond tab
            context_getc(context)
            # advance to next field
            context.state += 1
            break

        elif chars_stored < itemsize:
            # store value
            memory[memory_index] = context.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1

        # advance input stream
        context_getc(context)

    return 1


# cdef class SkipChromParser(Parser):
#     """Skip the CHROM field."""
#
#     cdef int parse(self) nogil except -1:
#         return skip_chrom(self.context)
#
#
# cdef inline int skip_chrom(ParserContext context) nogil except -1:
#     # TODO store chrom on context
#
#     while True:
#
#         if context.c == 0:
#             context.state = ParserState.EOF
#             return 0
#
#         elif context.c == LF or context.c == CR:
#             context.state = ParserState.EOL
#             return 0
#
#         elif context.c == TAB:
#             # advance input stream beyond tab
#             context_getc(context)
#             context.state += 1
#             return 1
#
#         # advance input stream
#         context_getc(context)


cdef class PosInt32Parser(Parser):
    """Parser for POS field."""

    cdef np.int32_t[:] memory

    cdef int parse(self) nogil except -1:
        return pos_parse(self.memory, self.context)

    def malloc(self):
        self.values = np.zeros(self.context.chunk_length, dtype='int32')
        self.memory = self.values
        self.memory[:] = -1

    def mkchunk(self, chunk, limit=None):
        chunk[POS_FIELD] = self.values[:limit]
        self.malloc()


cdef inline int pos_parse(integer[:] memory,
                          ParserContext context) nogil except -1:
    cdef:
        long value
        int success

    # debug('pos_parse', context)

    temp_clear(context)

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            break

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            break

        elif context.c == TAB:
            context_getc(context)
            context.state += 1
            break

        else:
            temp_append(context)

        # advance input stream
        context_getc(context)

    # parse string as integer
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index] = context.l

    return 1


cdef class SkipPosParser(Parser):
    """Skip the POS field."""

    cdef int parse(self) nogil except -1:
        return skip_pos(self.context)


cdef inline int skip_pos(ParserContext context) nogil except -1:
    # TODO put POS on context

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            return 0

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            return 0

        elif context.c == TAB:
            context_getc(context)
            context.state += 1
            return 1

        # advance input stream
        context_getc(context)


cdef class SkipFieldParser(Parser):
    """Skip a field."""

    cdef int parse(self) nogil except -1:
        return skip_field(self.context)


cdef inline int skip_field(ParserContext context) nogil except -1:

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            return 0

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            return 0

        elif context.c == TAB:
            context_getc(context)
            context.state += 1
            return 1

        # advance input stream
        context_getc(context)


cdef class SkipAllCalldataParser(Parser):
    """Skip a field."""

    cdef int parse(self) nogil except -1:
        return skip_all_calldata(self.context)


cdef inline int skip_all_calldata(ParserContext context) nogil except -1:

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            return 1

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            return 1

        # advance input stream
        context_getc(context)


cdef class AltParser(Parser):
    """Parser for ALT field."""

    cdef np.uint8_t[:] memory

    def __init__(self, ParserContext context, dtype, number):
        super(AltParser, self).__init__(context)
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number

    cdef int parse(self) nogil except -1:
        return alt_parse(self.memory, self.itemsize, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    def mkchunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values
        self.malloc()


cdef inline int alt_parse(np.uint8_t[:] memory,
                          int itemsize,
                          int number,
                          ParserContext context) nogil except -1:
    cdef:
        # index of alt values
        int alt_index = 0
        # index into memory view
        int memory_offset, memory_index
        # number of characters read into current value
        int chars_stored = 0

    # debug('alt_parse', context)

    # initialise memory offset and index
    memory_offset = context.chunk_variant_index * itemsize * number
    memory_index = memory_offset

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            break

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            break

        if context.c == TAB:
            context_getc(context)
            context.state += 1
            break

        elif context.c == COMMA:
            # advance value index
            alt_index += 1
            # set memory index to beginning of next item
            memory_index = memory_offset + (alt_index * itemsize)
            # reset chars stored
            chars_stored = 0

        elif chars_stored < itemsize and alt_index < number:
            # store value
            memory[memory_index] = context.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1

        # advance input stream
        context_getc(context)


cdef class QualFloat32Parser(Parser):

    cdef np.float32_t[:] memory

    def __init__(self, ParserContext context, fill):
        super(QualFloat32Parser, self).__init__(context)
        self.fill = fill

    cdef int parse(self) nogil except -1:
        return qual_parse(self.memory, self.context)

    def malloc(self):
        self.values = np.empty(self.context.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill

    def mkchunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]
        self.malloc()


cdef inline int qual_parse(floating[:] memory,
                           ParserContext context) nogil except -1:
    cdef:
        int success

    # debug('qual_parse', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == 0:
            context.state = ParserState.EOF
            break

        elif context.c == LF or context.c == CR:
            context.state = ParserState.EOL
            break

        elif context.c == TAB:
            # advance input stream beyond tab
            context_getc(context)
            context.state += 1
            break

        else:
            temp_append(context)

        # advance input stream
        context_getc(context)

    # parse string as floating
    success = temp_todouble(context)

    # store value
    if success:
        memory[context.chunk_variant_index] = context.d

    return 1


cdef class FilterParser(Parser):

    cdef np.uint8_t[:, :] memory

    def __init__(self, ParserContext context, filters):
        super(FilterParser, self).__init__(context)

        # setup filters
        context.filters = tuple(sorted(filters))
        # debug(filters, context)
        context.n_filters = len(context.filters)
        if context.filter_ptrs is not NULL:
            free(context.filter_ptrs)
        context.filter_ptrs = <char**> malloc(context.n_filters * sizeof(char*))
        for i in range(context.n_filters):
            context.filter_ptrs[i] = <char*> context.filters[i]

    cdef int parse(self) nogil except -1:
        return filter_parse(self, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_filters)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    def mkchunk(self, chunk, limit=None):
        for i, filter in enumerate(self.context.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]
        self.malloc()


cdef inline int filter_parse(FilterParser self,
                             ParserContext context) nogil except -1:
    cdef:
        int filter_index

    # debug('filter_parse: enter', context)

    # reset temporary buffer
    temp_clear(context)

    # check for explicit missing value
    if context.c == PERIOD:
        # debug('filter_parse: found missing value', context)

        while True:

            if context.c == 0:
                context.state = ParserState.EOF
                break

            elif context.c == LF or context.c == CR:
                context.state = ParserState.EOL
                break

            elif context.c == TAB:
                # advance input stream beyond tab
                context_getc(context)
                context.state += 1
                break

            # advance input stream
            context_getc(context)

        return 1

    while True:
        # debug('filter_parse: parsing filters', context)

        if context.c == 0:
            filter_store(self, context)
            context.state = ParserState.EOF
            break

        elif context.c == LF or context.c == CR:
            filter_store(self, context)
            context.state = ParserState.EOL
            break

        elif context.c == TAB:
            filter_store(self, context)
            # advance input stream beyond tab
            context_getc(context)
            context.state += 1
            break

        elif context.c == COMMA or context.c == COLON or context.c == SEMICOLON:
            # some of these delimiters are not strictly kosher, but have seen them
            filter_store(self, context)
            temp_clear(context)

        else:
            temp_append(context)

        # advance input stream
        context_getc(context)

    return 1


cdef inline int filter_store(FilterParser self,
                             ParserContext context) nogil except -1:
    cdef:
        int filter_index
        int i
        char* f

    if context.temp_size == 0:
        warn('empty FILTER', context)
        return 0

    temp_terminate(context)

    # search through filters to find index
    filter_index = search_sorted(context.temp, context.filter_ptrs, context.n_filters)

    # store value
    if filter_index >= 0:
        self.memory[context.chunk_variant_index, filter_index] = 1

    return 1


cdef class InfoParser(Parser):

    # cdef tuple infos
    # cdef dict parsers
    cdef Parser skip_parser

    def __init__(self, ParserContext context, infos, types, numbers):
        super(InfoParser, self).__init__(context)

        # setup info keys
        context.infos = tuple(sorted(infos))
        context.n_infos = len(context.infos)
        if context.info_ptrs is not NULL:
            free(context.info_ptrs)
        context.info_ptrs = <char**> malloc(context.n_infos * sizeof(char*))
        for i in range(context.n_infos):
            context.info_ptrs[i] = <char*> context.infos[i]

        # setup parsers
        context.info_parsers = list()
        self.skip_parser = SkipInfoFieldParser(context)
        for key in context.infos:
            t = types[key]
            n = numbers[key]
            if t == np.dtype(bool) or n == 0:
                parser = InfoFlagParser(context, key)
            elif t == np.dtype('int32'):
                parser = InfoInt32Parser(context, key, fill=-1, number=n)
            elif t == np.dtype('int64'):
                parser = InfoInt64Parser(context, key, fill=-1, number=n)
            elif t == np.dtype('float32'):
                parser = InfoFloat32Parser(context, key, fill=NAN, number=n)
            elif t == np.dtype('float64'):
                parser = InfoFloat64Parser(context, key, fill=NAN, number=n)
            elif t == np.dtype(bool):
                parser = InfoFlagParser(context, key)
            elif t.kind == 'S':
                parser = InfoStringParser(context, key, dtype=t, number=n)
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for INFO field %r, field will be '
                              'skipped' % (t, key))
            context.info_parsers.append(parser)

        # store pointers to parsers
        if context.info_parser_ptrs is not NULL:
            free(context.info_parser_ptrs)
        context.info_parser_ptrs = <PyObject**> malloc(context.n_infos *
                                                       sizeof(PyObject*))
        for i in range(context.n_infos):
            context.info_parser_ptrs[i] = <PyObject*> context.info_parsers[i]

    cdef int parse(self) nogil except -1:
        return info_parse(self, self.context)

    def malloc(self):
        for parser in self.context.info_parsers:
            parser.malloc()

    def mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.context.info_parsers:
            parser.mkchunk(chunk, limit=limit)


cdef inline int info_parse(InfoParser self,
                           ParserContext context) nogil except -1:


    # debug('info_parse', context)

    # check for explicit missing value
    if context.c == PERIOD:

        while True:

            if context.c == 0:
                context.state = ParserState.EOF
                return 0

            elif context.c == LF or context.c == CR:
                context.state = ParserState.EOL
                return 0

            elif context.c == TAB:
                # advance input stream beyond tab
                context_getc(context)
                context.state += 1
                return 1

            # advance input stream
            context_getc(context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == 0:
            info_store(self, context)
            context.state = ParserState.EOF
            return 0

        elif context.c == LF or context.c == CR:
            info_store(self, context)
            context.state = ParserState.EOL
            return 0

        elif context.c == TAB:
            info_store(self, context)
            # advance input stream beyond tab
            context_getc(context)
            context.state += 1
            return 1

        elif context.c == SEMICOLON:
            info_store(self, context)
            context_getc(context)

        elif context.c == EQUALS:
            # advance input stream beyond '='
            # TODO warn if empty key?
            context_getc(context)
            info_store(self, context)

        else:

            temp_append(context)
            context_getc(context)


cdef inline int search_sorted(char* query, char** compare, int n_items) nogil:
    cdef:
        int i

    # TODO smarter search

    # simple scan for now
    for i in range(n_items):
        if strcmp(query, compare[i]) == 0:
            return i

    return -1


cdef inline int info_store(InfoParser self,
                           ParserContext context) nogil except -1:
    cdef:
        int parser_index
        PyObject* parser

    if context.temp_size > 0:
        # treat temp as key

        # terminate temp
        temp_terminate(context)

        # search for index of current INFO key
        parser_index = search_sorted(context.temp, context.info_ptrs, context.n_infos)

        # clear out temp for good measure
        temp_clear(context)

        if parser_index >= 0:
            # obtain parser
            parser = context.info_parser_ptrs[parser_index]
            (<Parser> parser).parse()
        else:
            self.skip_parser.parse()

    # else:
    #
    #     warn('error parsing INFO field, missing key', context)
    #     # advance to next delimiter
    #     while context.c != SEMICOLON and \
    #             context.c != TAB and \
    #             context.c != LF and \
    #             context.c != CR and \
    #             context.c != 0:
    #         context_getc(context)
    #
    return 1


cdef class InfoParserBase(Parser):

    def __init__(self, ParserContext context, key, fill, number):
        super(InfoParserBase, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)
        self.fill = fill
        self.number = number

    def mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(<bytes>self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class InfoInt32Parser(InfoParserBase):

    cdef np.int32_t[:, :] memory

    cdef int parse(self) nogil except -1:
        return info_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class InfoInt64Parser(InfoParserBase):

    cdef np.int64_t[:, :] memory

    cdef int parse(self) nogil except -1:
        return info_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef inline int info_integer_parse(integer[:, :] memory,
                                   int number,
                                   ParserContext context) nogil except -1:
    cdef:
        int value_index = 0

    # debug('info_integer_parse', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == 0 or \
                context.c == LF or \
                context.c == CR or \
                context.c == TAB or \
                context.c == SEMICOLON:
            info_integer_store(memory, number, context, value_index)
            break

        elif context.c == COMMA:
            info_integer_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        else:
            temp_append(context)

        context_getc(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)

    return 1


cdef inline int info_integer_store(integer[:, :] memory,
                                   int number,
                                   ParserContext context,
                                   int value_index) nogil except -1:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return 1

    # parse string as integer
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, value_index] = context.l

    return 1


cdef class InfoFloat32Parser(InfoParserBase):

    cdef np.float32_t[:, :] memory

    cdef int parse(self) nogil except -1:
        return info_floating_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class InfoFloat64Parser(InfoParserBase):

    cdef np.float64_t[:, :] memory

    cdef int parse(self) nogil except -1:
        return info_floating_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef inline int info_floating_parse(floating[:, :] memory,
                                    int number,
                                    ParserContext context) nogil except -1:
    cdef:
        int value_index = 0

    # debug('info_floating_parse', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == 0 or \
                context.c == LF or \
                context.c == CR or \
                context.c == TAB or \
                context.c == SEMICOLON:
            info_floating_store(memory, number, context, value_index)
            break

        elif context.c == COMMA:
            info_floating_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        else:
            temp_append(context)

        context_getc(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)

    return 1


cdef inline int info_floating_store(floating[:, :] memory,
                                    int number,
                                    ParserContext context,
                                    int value_index) nogil except -1:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return 1

    # parse string as double
    success = temp_todouble(context)

    # store value
    if success:
        memory[context.chunk_variant_index, value_index] = context.d

    return 1


cdef class InfoFlagParser(Parser):

    cdef np.uint8_t[:] memory

    def __init__(self, ParserContext context, key):
        super(InfoFlagParser, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)

    cdef int parse(self) nogil except -1:
        # debug('InfoFlagParser.parse', self.context)
        self.memory[self.context.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while self.context.c != SEMICOLON and \
                self.context.c != TAB and \
                self.context.c != LF and \
                self.context.c != CR and \
                self.context.c != 0:
            context_getc(self.context)
        return 1

    def malloc(self):
        self.values = np.zeros(self.context.chunk_length, dtype='u1')
        self.memory = self.values

    def mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(<bytes>self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)
        self.malloc()


cdef class InfoStringParser(Parser):

    cdef np.uint8_t[:] memory

    def __init__(self, ParserContext context, key, dtype, number):
        super(InfoStringParser, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number

    cdef int parse(self) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # debug('InfoStringParser.parse', self.context)

        # initialise memory index
        memory_offset = self.context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        while True:

            if self.context.c == 0 or \
                    self.context.c == LF or \
                    self.context.c == CR or \
                    self.context.c == TAB or \
                    self.context.c == SEMICOLON:
                break

            elif self.context.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = self.context.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            context_getc(self.context)

        return 1

    def malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    def mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class FormatParser(Parser):

    cdef int parse(self) nogil except -1:
        return format_parse(self, self.context)


# break out method as function for profiling
cdef inline int format_parse(FormatParser self,
                             ParserContext context) nogil except -1:
    cdef:
        int format_index = 0
        int i

    # debug('format_parse: enter', context)

    # reset temporary buffer
    temp_clear(context)
    context.variant_n_formats = 0

    while True:

        if context.c == 0:
            # no point setting format
            context.state = ParserState.EOF
            return 0

        elif context.c == LF or context.c == CR:
            # no point setting format
            context.state = ParserState.EOL
            return 0

        elif context.c == TAB:
            # debug('format_parse: field end, setting', context)
            format_set(context, format_index)
            context.variant_n_formats = format_index + 1
            # we're done here, advance to next field
            context.state += 1
            context_getc(context)
            return 1

        elif context.c == COLON:
            # debug('format_parse: format end, setting', context)
            format_set(context, format_index)
            format_index += 1

        else:
            temp_append(context)

        # advance to next character
        context_getc(context)


cdef inline int format_set(ParserContext context,
                           int format_index) nogil except -1:
    cdef:
        PyObject* parser = NULL
        int parser_index
    # debug('format_set: enter', context)

    if format_index >= MAX_FORMATS:
        warn('MAX_FORMATS exceeded', context)
        return 0

    if context.temp_size == 0:
        warn('empty FORMAT', context)

    else:

        # terminate string
        temp_terminate(context)

        # find parser
        parser_index = search_sorted(context.temp, context.format_ptrs, context.n_formats)

        # set parser
        if parser_index >= 0:
            parser = context.calldata_parser_ptrs[parser_index]

    context.variant_calldata_parser_ptrs[format_index] = parser

    temp_clear(context)

    return 1


# noinspection PyShadowingBuiltins
cdef class CalldataParser(Parser):

    cdef tuple formats
    cdef Parser skip_parser

    def __init__(self, ParserContext context, formats, types, numbers):
        super(CalldataParser, self).__init__(context)
        self.skip_parser = SkipCalldataFieldParser(context)

        # setup formats
        context.formats = tuple(sorted(formats))
        context.n_formats = len(context.formats)
        if context.format_ptrs is not NULL:
            free(context.format_ptrs)
        context.format_ptrs = <char**> malloc(context.n_formats * sizeof(char*))
        for i in range(context.n_formats):
            context.format_ptrs[i] = <char*> context.formats[i]

        # setup parsers
        context.calldata_parsers = list()
        for key in context.formats:
            t = types[key]
            n = numbers[key]
            if key == b'GT' and t == np.dtype('int8'):
                parser = GenotypeInt8Parser(context, key, fill=-1)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = GenotypeInt16Parser(context, key, fill=-1)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = GenotypeInt32Parser(context, key, fill=-1)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = GenotypeInt64Parser(context, key, fill=-1)
            elif t == np.dtype('int8'):
                parser = CalldataInt8Parser(context, key, number=n, fill=-1)
            elif t == np.dtype('int16'):
                parser = CalldataInt16Parser(context, key, number=n, fill=-1)
            elif t == np.dtype('int32'):
                parser = CalldataInt32Parser(context, key, number=n, fill=-1)
            elif t == np.dtype('int64'):
                parser = CalldataInt64Parser(context, key, number=n, fill=-1)
            elif t == np.dtype('float32'):
                parser = CalldataFloat32Parser(context, key, number=n, fill=NAN)
            elif t == np.dtype('float64'):
                parser = CalldataFloat64Parser(context, key, number=n, fill=NAN)
            elif t.kind == 'S':
                parser = CalldataStringParser(context, key, dtype=t, number=n)
            # TODO unsigned int parsers
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for FORMAT field %r, field will be '
                              'skipped' % (t, key))
            context.calldata_parsers.append(parser)

        # store pointers for
        if context.calldata_parser_ptrs is not NULL:
            free(context.calldata_parser_ptrs)
        context.calldata_parser_ptrs = <PyObject**> malloc(context.n_formats *
                                                           sizeof(PyObject*))
        for i in range(context.n_formats):
            context.calldata_parser_ptrs[i] = <PyObject*> context.calldata_parsers[i]

    cdef int parse(self) nogil except -1:
        return calldata_parse(self, self.context)

    def malloc(self):
        cdef Parser parser
        for parser in self.context.calldata_parsers:
            parser.malloc()

    def mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.context.calldata_parsers:
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline int calldata_parse(CalldataParser self,
                               ParserContext context) nogil except -1:
    cdef:
        int i
        PyObject* parser

    # debug('calldata_parse: enter', context)

    # initialise context
    context.sample_index = 0
    context.format_index = 0

    while True:

        if context.c == 0:
            # debug('calldata_parse: EOF', context)
            context.state = ParserState.EOF
            return 1

        elif context.c == LF or context.c == CR:
            # debug('calldata_parse: EOL', context)
            context.state = ParserState.EOL
            context.sample_index = 0
            return 1

        elif context.c == TAB:
            # debug('calldata_parse: TAB', context)
            context.sample_index += 1
            context.format_index = 0
            context_getc(context)

        elif context.c == COLON:
            # debug('calldata_parse: COLON', context)
            context.format_index += 1
            context_getc(context)

        elif context.format_index < MAX_FORMATS:
            # debug('calldata_parse: parse', context)

            parser = context.variant_calldata_parser_ptrs[context.format_index]
            if parser is NULL:
                self.skip_parser.parse()

            else:
                # jump through some hoops to avoid references (which need the GIL)
                (<Parser>parser).parse()

        else:
            # debug('calldata_parse: exceeded MAX_FORMATS', context)
            self.skip_parser.parse()


cdef class SkipInfoFieldParser(Parser):

    cdef int parse(self) nogil except -1:
        while self.context.c != SEMICOLON and \
                self.context.c != TAB and \
                self.context.c != CR and \
                self.context.c != LF and \
                self.context.c != 0:
            context_getc(self.context)
        return 1


cdef class SkipCalldataFieldParser(Parser):

    cdef int parse(self) nogil except -1:
        while self.context.c != COLON and \
                self.context.c != TAB and \
                self.context.c != CR and \
                self.context.c != LF and \
                self.context.c != 0:
            context_getc(self.context)
        return 1


cdef inline int calldata_integer_parse(integer[:, :, :] memory,
                                       int number,
                                       ParserContext context) nogil except -1:
    cdef:
        int value_index = 0

    # debug('calldata_integer_parse: enter', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            # debug('calldata_integer_parse: COMMA', context)
            calldata_integer_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or \
                context.c == TAB or \
                context.c == LF or \
                context.c == CR or \
                context.c == 0:
            # debug('calldata_integer_parse: DELIMITER', context)
            calldata_integer_store(memory, number, context, value_index)
            break

        else:
            # debug('calldata_integer_parse: temp_append', context)
            temp_append(context)

        # debug('calldata_integer_parse: getc', context)
        context_getc(context)

    # debug('calldata_integer_parse: exit', context)
    return 1


cdef inline int calldata_integer_store(integer[:, :, :] memory,
                                       int number,
                                       ParserContext context,
                                       int value_index) nogil except -1:
    cdef:
        int success

    # debug('calldata_integer_store: enter', context)

    if value_index >= number:
        # more values than we have room for, ignore
        return 1

    # debug('parse string as integer', context)
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, value_index] = context.l

    # debug('calldata_integer_store: exit', context)
    return 1


cdef inline int calldata_floating_parse(floating[:, :, :] memory,
                                        int number,
                                        ParserContext context) nogil except -1:
    cdef:
        int value_index = 0

    # debug('calldata_floating_parse: enter', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            calldata_floating_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or \
                context.c == TAB or \
                context.c == LF or \
                context.c == CR or \
                context.c == 0:
            calldata_floating_store(memory, number, context, value_index)
            return 1

        else:
            temp_append(context)

        context_getc(context)


cdef inline int calldata_floating_store(floating[:, :, :] memory,
                                        int number,
                                        ParserContext context,
                                        int value_index) nogil except -1:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return 1

    # parse string as floating
    success = temp_todouble(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, value_index] = context.d

    return 1


cdef class GenotypeParserBase(Parser):

    def __init__(self, ParserContext context, bytes key, fill):
        # debug('GenotypeParserBase.__init__: enter', context)
        super(GenotypeParserBase, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)
        self.fill = fill

    def mkchunk(self, chunk, limit=None):
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt8Parser(GenotypeParserBase):

    cdef np.int8_t[:, :, :] memory
    # TODO cdef object dtype = 'int8' ... can factor out malloc?

    def malloc(self):
        # debug('GenotypeInt8Parser.malloc: enter', self.context)
        shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
        self.values = np.empty(shape, dtype='int8')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef int parse(self) nogil except -1:
        # debug('GenotypeInt8Parser.parse: enter', self.context)
        return genotype_parse(self.memory, self.context)


cdef class GenotypeInt16Parser(GenotypeParserBase):

    cdef np.int16_t[:, :, :] memory

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
        self.values = np.empty(shape, dtype='int16')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef int parse(self) nogil except -1:
        return genotype_parse(self.memory, self.context)


cdef class GenotypeInt32Parser(GenotypeParserBase):

    cdef np.int32_t[:, :, :] memory

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
        self.values = np.empty(shape, dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef int parse(self) nogil except -1:
        return genotype_parse(self.memory, self.context)


cdef class GenotypeInt64Parser(GenotypeParserBase):

    cdef np.int64_t[:, :, :] memory

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.context.ploidy)
        self.values = np.empty(shape, dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef int parse(self) nogil except -1:
        return genotype_parse(self.memory, self.context)


cdef inline int genotype_parse(integer[:, :, :] memory,
                               ParserContext context) nogil except -1:
    cdef:
        int allele_index = 0

    # debug('genotype_parse: enter', context)

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == SLASH or context.c == PIPE:
            genotype_store(memory, context, allele_index)
            allele_index += 1
            temp_clear(context)

        elif context.c == COLON or \
                context.c == TAB or \
                context.c == LF or \
                context.c == CR or \
                context.c == 0:
            genotype_store(memory, context, allele_index)
            return 1

        else:
            temp_append(context)

        context_getc(context)


cdef inline int genotype_store(integer[:, :, :] memory,
                               ParserContext context,
                               int allele_index) nogil except -1:
    cdef:
        int success

    # debug('genotype_store: enter', context)

    if allele_index >= context.ploidy:
        # more alleles than we've made room for, ignore
        return 0

    # attempt to parse allele
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, allele_index] = \
            context.l

    return 1


cdef class CalldataParserBase(Parser):

    def __init__(self, ParserContext context, bytes key, fill, number):
        super(CalldataParserBase, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)
        self.number = number
        self.fill = fill

    def mkchunk(self, chunk, limit=None):
        field = 'calldata/' + str(<bytes>self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values
        self.malloc()


cdef class CalldataInt8Parser(CalldataParserBase):

    cdef np.int8_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='int8')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt16Parser(CalldataParserBase):

    cdef np.int16_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='int16')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt32Parser(CalldataParserBase):

    cdef np.int32_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt64Parser(CalldataParserBase):

    cdef np.int64_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_integer_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill


# TODO unsigned int calldata parsers


cdef class CalldataFloat32Parser(CalldataParserBase):

    cdef np.float32_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_floating_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataFloat64Parser(CalldataParserBase):

    cdef np.float64_t[:, :, :] memory

    cdef int parse(self) nogil except -1:
        return calldata_floating_parse(self.memory, self.number, self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.empty(shape, dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataStringParser(Parser):

    cdef np.uint8_t[:] memory

    def __init__(self, ParserContext context, key, dtype, number):
        super(CalldataStringParser, self).__init__(context)
        self.key = PyBytes_AS_STRING(key)
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number

    cdef int parse(self) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # debug('CalldataStringParser.parse: enter', self.context)

        # initialise memory index
        memory_offset = ((self.context.chunk_variant_index *
                         self.context.n_samples *
                         self.number *
                         self.itemsize) +
                         (self.context.sample_index *
                          self.number *
                          self.itemsize))
        memory_index = memory_offset

        # read characters until tab
        while True:

            if self.context.c == TAB or \
                    self.context.c == COLON or \
                    self.context.c == CR or \
                    self.context.c == LF or \
                    self.context.c == 0:
                return 1

            elif self.context.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = self.context.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1

            # advance input stream
            context_getc(self.context)

    def malloc(self):
        shape = (self.context.chunk_length, self.context.n_samples, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    def mkchunk(self, chunk, limit=None):
        field = 'calldata/' + str(<bytes>self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values
        self.malloc()


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
