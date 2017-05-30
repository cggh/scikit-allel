# cython: language_level=3
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
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
from libc.stdlib cimport strtol, strtof, strtod, malloc, free
import numpy as np
cimport numpy as np
import cython
# noinspection PyUnresolvedReferences
cimport cython


cdef double NAN = np.nan


from cpython.ref cimport PyObject
from cpython.list cimport PyList_GET_ITEM
cdef extern from "Python.h":
    char* PyByteArray_AS_STRING(object string)


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
    np.float32_t
    np.float64_t


cdef char TAB = b'\t'
cdef char NEWLINE = b'\n'
cdef char HASH = b'#'
cdef char COLON = b':'
cdef char SEMICOLON = b';'
cdef char PERIOD = b'.'
cdef char COMMA = b','
cdef char SLASH = b'/'
cdef char PIPE = b'|'
cdef char EQUALS = b'='


CHROM_FIELD = 'variants/CHROM'
POS_FIELD = 'variants/POS'
ID_FIELD = 'variants/ID'
REF_FIELD = 'variants/REF'
ALT_FIELD = 'variants/ALT'
QUAL_FIELD = 'variants/QUAL'


cdef void warn(message, ParserContext context) nogil:
    with gil:
        # TODO customize message based on state (CHROM, POS, etc.)
        message += '; variant index: %s' % context.variant_index
        b = PyBytes_FromStringAndSize(context.temp, context.temp_size)
        message += '; temporary buffer: %s' % b
        warnings.warn(message)


def iter_vcf(input_file, int input_buffer_size, int chunk_length, int temp_buffer_size,
             headers, fields, types, numbers, ploidy=2):
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

    # setup output
    # TODO yield chunks
    chunks = []

    # setup context
    n_samples = len(headers.samples)
    context = ParserContext(input_file=input_file,
                        input_buffer_size=input_buffer_size,
                        temp_buffer_size=temp_buffer_size,
                        n_samples=n_samples,
                        chunk_length=chunk_length,
                        ploidy=ploidy)

    # copy so we don't modify someone else's data
    fields = set(fields)

    # setup CHROM parser
    if CHROM_FIELD in fields:
        chrom_parser = StringParser(field=CHROM_FIELD, chunk_length=chunk_length,
                                    dtype=types[CHROM_FIELD])
        fields.remove(CHROM_FIELD)
    else:
        chrom_parser = SkipChromParser()

    # setup POS parser
    if POS_FIELD in fields:
        # TODO user-provided type
        pos_parser = PosInt32Parser(chunk_length=chunk_length)
        fields.remove(POS_FIELD)
    else:
        pos_parser = SkipPosParser()

    # setup ID parser
    if ID_FIELD in fields:
        id_parser = StringParser(field=ID_FIELD, chunk_length=chunk_length,
                                 dtype=types[ID_FIELD])
        fields.remove(ID_FIELD)
    else:
        id_parser = SkipFieldParser()

    # setup REF parser
    if REF_FIELD in fields:
        ref_parser = StringParser(field=REF_FIELD, chunk_length=chunk_length,
                                  dtype=types[REF_FIELD])
        fields.remove(REF_FIELD)
    else:
        ref_parser = SkipFieldParser()

    # setup ALT parser
    if ALT_FIELD in fields:
        t = types[ALT_FIELD]
        n = numbers[ALT_FIELD]
        alt_parser = AltParser(chunk_length=chunk_length, dtype=t, number=n)
        fields.remove(ALT_FIELD)
    else:
        alt_parser = SkipFieldParser()

    # setup QUAL parser
    if QUAL_FIELD in fields:
        # TODO user-provided type
        qual_parser = QualFloat32Parser(chunk_length=chunk_length, fill=-1)
        fields.remove(QUAL_FIELD)
    else:
        qual_parser = SkipFieldParser()

    # setup FILTER parser
    filters = list()
    for field in list(fields):
        if field.startswith('variants/FILTER_'):
            filter = field[16:].encode('ascii')
            filters.append(filter)
            fields.remove(field)
    if filters:
        filter_parser = FilterParser(chunk_length=chunk_length, filters=filters)
    else:
        filter_parser = SkipFieldParser()

    # setup INFO parsers
    infos = list()
    info_types = dict()
    info_numbers = dict()
    # assume any variants fields left are INFO
    for field in list(fields):
        group, name = field.split('/')
        if group == 'variants':
            key = name.encode('ascii')
            infos.append(key)
            fields.remove(field)
            info_types[key] = types[field]
            info_numbers[key] = numbers[field]
    if infos:
        info_parser = InfoParser(chunk_length=chunk_length, infos=infos, types=info_types,
                                 numbers=info_numbers)
    else:
        info_parser = SkipFieldParser()

    # setup FORMAT and calldata parsers
    formats = list()
    format_types = dict()
    format_numbers = dict()
    for field in list(fields):
        group, name = field.split('/')
        if group == 'calldata':
            key = name.encode('ascii')
            formats.append(key)
            fields.remove(field)
            format_types[key] = types[field]
            format_numbers[key] = numbers[field]
    if formats:
        format_parser = FormatParser()
        calldata_parser = CalldataParser(chunk_length=chunk_length,
                                         formats=formats, types=format_types,
                                         numbers=format_numbers,
                                         n_samples=context.n_samples,
                                         ploidy=2)
    else:
        format_parser = SkipFieldParser()
        calldata_parser = SkipAllCalldataParser()

    if fields:
        # shouldn't ever be any left over
        raise RuntimeError('unexpected fields left over: %r' % set(fields))

    with nogil:

        while True:

            if context.c == 0:
                break

            elif context.state == ParserState.CHROM:
                chrom_parser.parse(context)
                context.state = ParserState.POS

            elif context.state == ParserState.POS:
                pos_parser.parse(context)
                context.state = ParserState.ID

            elif context.state == ParserState.ID:
                id_parser.parse(context)
                context.state = ParserState.REF

            elif context.state == ParserState.REF:
                ref_parser.parse(context)
                context.state = ParserState.ALT

            elif context.state == ParserState.ALT:
                alt_parser.parse(context)
                context.state = ParserState.QUAL

            elif context.state == ParserState.QUAL:
                qual_parser.parse(context)
                context.state = ParserState.FILTER

            elif context.state == ParserState.FILTER:
                filter_parser.parse(context)
                context.state = ParserState.INFO

            elif context.state == ParserState.INFO:
                # debug(context.variant_index, 'parse INFO')
                info_parser.parse(context)
                context.state = ParserState.FORMAT

            elif context.state == ParserState.FORMAT:
                format_parser.parse(context)
                context.state = ParserState.CALLDATA

            elif context.state == ParserState.CALLDATA:
                calldata_parser.parse(context)
                context.state = ParserState.CHROM

                # setup next variant
                # debug('setup next variant')
                context.variant_index += 1
                if context.chunk_variant_index < chunk_length - 1:
                    context.chunk_variant_index += 1

                else:

                    with gil:

                        # build chunk for output
                        chunk = dict()
                        chrom_parser.mkchunk(chunk)
                        pos_parser.mkchunk(chunk)
                        id_parser.mkchunk(chunk)
                        ref_parser.mkchunk(chunk)
                        alt_parser.mkchunk(chunk)
                        qual_parser.mkchunk(chunk)
                        filter_parser.mkchunk(chunk)
                        info_parser.mkchunk(chunk)
                        calldata_parser.mkchunk(chunk)
                        # TODO yield
                        chunks.append(chunk)

                    # setup next chunk
                    context.chunk_variant_index = 0

            else:

                with gil:
                    # shouldn't ever happen
                    raise RuntimeError('unexpected parser state')

    # left-over chunk
    limit = context.chunk_variant_index
    if limit > 0:
        chunk = dict()
        chrom_parser.mkchunk(chunk, limit=limit)
        pos_parser.mkchunk(chunk, limit=limit)
        id_parser.mkchunk(chunk, limit=limit)
        ref_parser.mkchunk(chunk, limit=limit)
        alt_parser.mkchunk(chunk, limit=limit)
        qual_parser.mkchunk(chunk, limit=limit)
        filter_parser.mkchunk(chunk, limit=limit)
        info_parser.mkchunk(chunk, limit=limit)
        calldata_parser.mkchunk(chunk, limit=limit)
        # TODO yield
        chunks.append(chunk)

    # TODO yield
    return chunks


def debug(*msg):
    print(*msg, file=sys.stderr)
    sys.stderr.flush()


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
    CALLDATA


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
        # list formats
        int n_formats
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

        # initialize state
        self.state = ParserState.CHROM
        self.n_samples = n_samples
        self.variant_index = 0
        self.chunk_variant_index = 0
        self.sample_index = 0
        self.format_index = 0
        self.calldata_parser_ptrs = NULL
        self.variant_calldata_parser_ptrs = NULL
        self.chunk_length = chunk_length
        self.ploidy = ploidy

    def __dealloc__(self):
        if self.calldata_parser_ptrs is not NULL:
            free(self.calldata_parser_ptrs)
        if self.variant_calldata_parser_ptrs is not NULL:
            free(self.variant_calldata_parser_ptrs)


cdef inline int context_fill_buffer(ParserContext context) nogil except -1:
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


cdef inline int context_getc(ParserContext context) nogil except -1:

    if context.input is context.input_end:
        context_fill_buffer(context)

    if context.input is NULL:
        context.c = 0
        return 0

    else:
        context.c = context.input[0]
        context.input += 1
        return 1


cdef inline void temp_clear(ParserContext context) nogil:
    context.temp_size = 0


cdef inline int temp_append(ParserContext context) nogil except -1:

    # if context.temp_size >= context.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # store current character
    context.temp[context.temp_size] = context.c

    # increase size
    context.temp_size += 1

    return 1


cdef inline int temp_tolong(ParserContext context) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if context.temp_size == 0:

        warn('expected integer, found empty value', context)
        return 0

    if context.temp_size == 1 and context.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # if context.temp_size >= context.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # terminate string
    context.temp[context.temp_size] = 0

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


cdef inline int temp_todouble(ParserContext context) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if context.temp_size == 0:

        warn('expected floating, found empty value', context)
        return 0

    if context.temp_size == 1 and context.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # if context.temp_size >= context.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # terminate string
    context.temp[context.temp_size] = 0

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

    cdef object values
    cdef ParserContext context

    cdef void parse(self) nogil:
        pass

    cdef void malloc(self):
        pass

    cdef void mkchunk(self, chunk, limit=None):
        pass


def check_string_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != 'S':
        raise ValueError('expected byte string ("S") dtype, found: %r' % dtype)
    return dtype


cdef class StringParser(Parser):
    """Generic string field parser, used for CHROM, ID, REF."""

    cdef object dtype
    cdef int itemsize
    cdef np.uint8_t[:] memory
    cdef object field

    def __cinit__(self, context, field, dtype):
        self.context = context
        self.field = field
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros(self.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    cdef void parse(self) nogil:
        StringParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void StringParser_parse(StringParser self, ParserContext context) nogil:
    cdef:
        # index into memory view
        int memory_index
        # number of characters read into current value
        int chars_stored = 0

    # debug('StringParser_parse', self.field)

    # initialise memory index
    memory_index = context.chunk_variant_index * self.itemsize

    # read characters until tab
    while context.c != TAB:
        if chars_stored < self.itemsize:
            # store value
            self.memory[memory_index] = context.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context_getc(context)

    # advance input stream beyond tab
    context_getc(context)

    # debug(context.variant_index, self.field, self.values[context.chunk_variant_index],
    #       chars_stored)


cdef class SkipChromParser(Parser):
    """Skip the CHROM field."""

    def __cinit__(self, ParserContext context):
        self.context = context

    cdef void malloc(self):
        pass

    cdef void parse(self) nogil:
        SkipChromParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipChromParser_parse(SkipChromParser self, ParserContext context) nogil:

    # TODO store chrom on context

    # read characters until tab
    while context.c != TAB:
        context_getc(context)

    # advance input stream beyond tab
    context_getc(context)


cdef class PosInt32Parser(Parser):
    """Parser for POS field."""

    cdef np.int32_t[:] memory

    def __cinit__(self, ParserContext context, chunk_length):
        self.context = context
        self.chunk_length = chunk_length
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros(self.chunk_length, dtype='int32')
        self.memory = self.values
        self.memory[:] = -1

    cdef void parse(self) nogil:
        PosInt32Parser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[POS_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void PosInt32Parser_parse(PosInt32Parser self, ParserContext context) nogil:
    cdef:
        long value
        int success

    # reset temporary buffer
    temp_clear(context)

    # read into temporary buffer until tab
    while context.c != TAB:
        temp_append(context)
        context_getc(context)

    # parse string as integer
    success = temp_tolong(context)

    # store value
    if success:
        self.memory[context.chunk_variant_index] = context.l

    # advance input stream
    context_getc(context)


cdef class SkipPosParser(Parser):
    """Skip the POS field."""

    def __cinit__(self, ParserContext context):
        self.context = context

    cdef void malloc(self):
        pass

    cdef void parse(self) nogil:
        SkipPosParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipPosParser_parse(SkipPosParser self, ParserContext context) nogil:

    # TODO store pos on context

    # read characters until tab
    while context.c != TAB:
        context_getc(context)

    # advance input stream beyond tab
    context_getc(context)


cdef class SkipFieldParser(Parser):
    """Skip a field."""

    def __cinit__(self, ParserContext context):
        self.context = context

    cdef void malloc(self):
        pass

    cdef void parse(self) nogil:
        SkipFieldParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef inline int SkipFieldParser_parse(SkipFieldParser self,
                                      ParserContext context) nogil except -1:

    # read characters until tab or newline
    while self.context.c != TAB and self.context.c != NEWLINE and self.context.c != 0:
        context_getc(self.context)

    # advance input stream beyond tab or newline
    context_getc(self.context)


cdef class SkipAllCalldataParser(Parser):
    """Skip a field."""

    def __cinit__(self, ParserContext context):
        self.context = context

    cdef void malloc(self):
        pass

    cdef void parse(self) nogil:
        # read characters until newline
        while self.context.c != NEWLINE and self.context.c != 0:
            context_getc(self.context)

        # advance input stream beyond newline
        context_getc(self.context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef class AltParser(Parser):
    """Parser for ALT field."""

    cdef object dtype
    cdef int itemsize
    cdef int number
    cdef np.uint8_t[:] memory

    def __cinit__(self, ParserContext context, chunk_length, dtype, number):
        self.context = context
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number
        self.malloc()

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    cdef void parse(self) nogil:
        AltParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values
        self.malloc()


# break out method as function for profiling
cdef inline void AltParser_parse(AltParser self, ParserContext context) nogil:
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

    # read characters until tab
    while True:
        if context.c == TAB:
            context_getc(context)
            break
        elif context.c == COMMA:
            # advance value index
            alt_index += 1
            # set memory index to beginning of next item
            memory_index = memory_offset + (alt_index * self.itemsize)
            # reset chars stored
            chars_stored = 0
        elif chars_stored < self.itemsize and alt_index < self.number:
            # store value
            self.memory[memory_index] = context.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context_getc(context)

    # debug(context.variant_index, 'ALT', self.values[context.chunk_variant_index])


cdef class QualFloat32Parser(Parser):

    cdef np.float32_t fill
    cdef np.float32_t[:] memory

    def __cinit__(self, ParserContext context, fill):
        self.context = context
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        self.values = np.empty(self.context.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self) nogil:
        QualFloat32Parser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void QualFloat32Parser_parse(QualFloat32Parser self, ParserContext context) nogil:
    cdef:
        int success

    # reset temporary buffer
    temp_clear(context)

    # read into temporary buffer until tab
    while context.c != TAB:
        temp_append(context)
        context_getc(context)

    # parse string as floating
    success = temp_todouble(context)

    # store value
    if success:
        self.memory[context.chunk_variant_index] = context.d

    # advance input stream
    context_getc(context)


cdef class FilterParser(Parser):

    cdef tuple filters
    cdef dict filter_position
    cdef np.uint8_t[:, :] memory

    def __cinit__(self, ParserContext context, filters):
        self.context = context
        self.filters = tuple(filters)
        self.filter_position = {f: i for i, f in enumerate(self.filters)}
        self.malloc()

    cdef void malloc(self):
        shape = (self.context.chunk_length, len(self.filters) + 1)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    cdef void parse(self) nogil:
        FilterParser_parse(self)

    cdef void mkchunk(self, chunk, limit=None):
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]
        self.malloc()


# break out method as function for profiling
cdef inline void FilterParser_parse(FilterParser self, ParserContext context) nogil:
    cdef:
        int filter_index

    # reset temporary buffer
    temp_clear(context)

    # check for explicit missing value
    if context.c == PERIOD:
        while context.c != TAB:
            context_getc(context)
        context_getc(context)
        return

    while True:

        if context.c == TAB or context.c == NEWLINE or context.c == 0:
            FilterParser_store(self, context)
            break

        elif context.c == COMMA or context.c == COLON or context.c == SEMICOLON:
            # some of these delimiters are not strictly kosher, but have seen them
            FilterParser_store(self, context)
            temp_clear(context)

        else:
            temp_append(context)

        # advance to next character
        context_getc(context)

    # advance to next field
    context_getc(context)

    # debug(context.variant_index, 'FILTER', self.values[context.chunk_variant_index])


cdef inline void FilterParser_store(FilterParser self, ParserContext context) nogil:
    cdef:
        int filter_index

    if context.temp_size == 0:
        warn('empty FILTER', context)
        return

    # TODO nogil version?

    with gil:

        # read filter into byte string
        f = PyBytes_FromStringAndSize(context.temp, context.temp_size)

        # find filter position
        filter_index = self.filter_position.get(f, -1)

    # store value
    if filter_index >= 0:
        self.memory[context.chunk_variant_index, filter_index] = 1

    # debug(context.variant_index, 'FILTER', f)


cdef class InfoParser(Parser):

    cdef tuple infos
    cdef dict parsers
    cdef Parser skip_parser

    def __cinit__(self, ParserContext context, infos, types, numbers):
        self.context = context
        self.infos = tuple(infos)
        self.parsers = dict()
        self.skip_parser = SkipInfoFieldParser()
        for key in self.infos:
            t = types[key]
            n = numbers[key]
            # debug('setting up INFO parser', key, t, n)
            if t == np.dtype(bool) or n == 0:
                parser = InfoFlagParser(key, chunk_length=chunk_length)
            elif t == np.dtype('int32'):
                parser = InfoInt32Parser(key, fill=-1, chunk_length=chunk_length,
                                              number=n)
            elif t == np.dtype('int64'):
                parser = InfoInt64Parser(key, fill=-1, chunk_length=chunk_length,
                                              number=n)
            elif t == np.dtype('float32'):
                parser = InfoFloat32Parser(key, fill=NAN,
                                                chunk_length=chunk_length,
                                                number=n)
            elif t == np.dtype('float64'):
                parser = InfoFloat64Parser(key, fill=NAN,
                                                chunk_length=chunk_length,
                                                number=n)
            elif t == np.dtype(bool):
                parser = InfoFlagParser(key, chunk_length=chunk_length)
            elif t.kind == 'S':
                parser = InfoStringParser(key, chunk_length=chunk_length,
                                               dtype=t, number=n)
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for INFO field %r, field will be '
                              'skipped' % (t, key))
            self.parsers[key] = parser

    cdef void parse(self) nogil:
        # debug(context.variant_index, 'InfoParser.parse')
        InfoParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void InfoParser_parse(InfoParser self, ParserContext context) nogil:
    # cdef:
    #     Parser parser

    # debug(context.variant_index, 'InfoParser_parse')

    # check for explicit missing value
    if context.c == PERIOD:
        while context.c != TAB:
            context_getc(context)
        context_getc(context)
        return

    # reset temporary buffer
    temp_clear(context)

    with gil:
        # TODO nogil version?

        while True:

            if context.c == TAB or context.c == NEWLINE or context.c == 0:
                # debug(context.variant_index, 'end of INFO')
                # handle flags
                if context.temp_size > 0:
                    key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                    (<Parser>self.parsers.get(key, self.skip_parser)).parse(context)
                break

            elif context.c == EQUALS:
                # debug(context.variant_index, 'INFO =')
                context_getc(context)
                if context.temp_size > 0:
                    key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                    # debug(context.variant_index, 'INFO parsing value for key', key)
                    (<Parser>self.parsers.get(key, self.skip_parser)).parse(context)
                    temp_clear(context)
                else:
                    warn('error parsing INFO field, missing key', context)
                    # advance to next sub-field
                    while context.c != TAB and context.c != SEMICOLON and context.c != 0:
                        context_getc(context)

            elif context.c == SEMICOLON:
                # debug(context.variant_index, 'end of INFO subfield')
                # handle flags
                if context.temp_size > 0:
                    key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                    (<Parser>self.parsers.get(key, self.skip_parser)).parse(context)
                    # debug(context.variant_index, 'detected flag', key, parser)
                    # parser.parse(context)
                    temp_clear(context)
                context_getc(context)

            else:

                # debug(context.variant_index, 'storing INFO key character', bytes([context.c]))
                temp_append(context)
                context_getc(context)

    # advance to next field
    context_getc(context)


cdef class InfoParserBase(Parser):

    cdef char* key
    cdef int number
    cdef object fill

    def __cinit__(self, ParserContext context, key, fill, number):
        self.context = context
        self.key = PyBytes_AS_STRING(key)
        self.fill = fill
        self.number = number
        self.malloc()

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(<bytes>self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class InfoInt32Parser(InfoParserBase):

    cdef np.int32_t[:, :] memory

    cdef void parse(self) nogil:
        info_integer_parse(self.memory, self.number, self.context)

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class InfoInt64Parser(InfoParserBase):

    cdef np.int64_t[:, :] memory

    cdef void parse(self) nogil:
        info_integer_parse(self.memory, self.number, self.context)

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef inline void info_integer_parse(integer[:, :] memory,
                                    int number,
                                    ParserContext context) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:

            info_integer_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == SEMICOLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            info_integer_store(memory, number, context, value_index)
            break

        else:

            temp_append(context)

        context_getc(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void info_integer_store(integer[:, :] memory,
                                    int number,
                                    ParserContext context,
                                    int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, value_index] = context.l


cdef class InfoFloat32Parser(InfoParserBase):

    cdef np.float32_t[:, :] memory

    cdef void parse(self) nogil:
        info_floating_parse(self.memory, self.number, self.context)

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class InfoFloat64Parser(InfoParserBase):

    cdef np.float64_t[:, :] memory

    cdef void parse(self) nogil:
        info_floating_parse(self.memory, self.number, self.context)

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.empty(shape, dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef inline void info_floating_parse(floating[:, :] memory,
                                     int number,
                                     ParserContext context) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            info_floating_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == SEMICOLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            info_floating_store(memory, number, context, value_index)
            break

        else:
            temp_append(context)

        context_getc(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void info_floating_store(floating[:, :] memory,
                                     int number,
                                     ParserContext context,
                                     int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as double
    success = temp_todouble(context)

    # store value
    if success:
        memory[context.chunk_variant_index, value_index] = context.d


# TODO continue here
cdef class InfoFlagParser(Parser):

    cdef np.uint8_t[:] memory
    cdef char* key

    def __cinit__(self, ParserContext context, key, chunk_length):
        self.context = context
        self.key = PyBytes_AS_STRING(key)
        self.malloc()

    cdef void parse(self) nogil:
        self.memory[self.context.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while self.context.c != SEMICOLON and \
                self.context.c != TAB and \
                self.context.c != NEWLINE and \
                self.context.c != 0:
            context_getc(self.context)

    cdef void malloc(self):
        self.values = np.zeros(self.context.chunk_length, dtype='u1')
        self.memory = self.values

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)
        self.malloc()


cdef class InfoStringParser(InfoParserBase):

    cdef object dtype
    cdef int itemsize

    def __cinit__(self, ParserContext context, key, dtype, number):
        self.context = context
        self.key = PyBytes_AS_STRING(key)
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number
        self.malloc()

    cdef void malloc(self):
        shape = (self.context.chunk_length, self.number)
        self.values = np.zeros(shape, dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    cdef void parse(self) nogil:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = self.context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        # read characters until tab
        while True:
            if self.context.c == TAB or self.context.c == SEMICOLON:
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

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class FormatParser(Parser):

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self) nogil:
        FormatParser_parse(self, self.context)


# break out method as function for profiling
cdef inline void FormatParser_parse(FormatParser self, ParserContext context) nogil:
    cdef:
        char* format

    # debug('FormatParser_parse()')

    # reset temporary buffer
    temp_clear(context)

    with gil:

        # TODO nogil version

        context.formats = []

        while True:
            # debug(context.c)

            if context.c == TAB or context.c == NEWLINE:

                # add last format
                if context.temp_size > 0:
                    # debug('add last format')
                    f = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                    context.formats.append(f)

                # we're done here
                break

            elif context.c == COLON:

                # add format
                if context.temp_size > 0:
                    # debug('add format')
                    f = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                    context.formats.append(f)
                    temp_clear(context)

            else:
                # debug('other')

                temp_append(context)

            # advance to next character
            context_getc(context)

        context.n_formats = len(context.formats)

    # advance to next field
    context_getc(context)

    # debug(context.variant_index, 'FORMAT', formats)

# noinspection PyShadowingBuiltins
cdef class CalldataParser(Parser):

    cdef tuple formats
    cdef dict parsers
    cdef Parser skip_parser

    def __cinit__(self, chunk_length, formats, types, numbers, n_samples, ploidy,
                  ParserContext context):
        self.chunk_length = chunk_length
        self.formats = tuple(formats)
        self.parsers = dict()
        self.skip_parser = SkipCalldataFieldParser()
        context.calldata_parsers = list()
        # TODO calldata parser pointers
        for key in formats:
            t = types[key]
            n = numbers[key]
            if key == b'GT' and t == np.dtype('int8'):
                parser = GenotypeInt8Parser(key, chunk_length=chunk_length,
                                            n_samples=n_samples,
                                            ploidy=ploidy,
                                            fill=-1)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = GenotypeInt16Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             ploidy=ploidy,
                                             fill=-1)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = GenotypeInt32Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             ploidy=ploidy,
                                             fill=-1)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = GenotypeInt64Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             ploidy=ploidy,
                                             fill=-1)
            elif t == np.dtype('int8'):
                parser = CalldataInt8Parser(key, chunk_length=chunk_length,
                                            n_samples=n_samples,
                                            number=n,
                                            fill=-1)
            elif t == np.dtype('int16'):
                parser = CalldataInt16Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             number=n,
                                             fill=-1)
            elif t == np.dtype('int32'):
                parser = CalldataInt32Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             number=n,
                                             fill=-1)
            elif t == np.dtype('int64'):
                parser = CalldataInt64Parser(key, chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             number=n,
                                             fill=-1)
            elif t == np.dtype('float32'):
                parser = CalldataFloat32Parser(key, chunk_length=chunk_length,
                                               n_samples=n_samples,
                                               number=n,
                                               fill=NAN)
            elif t == np.dtype('float64'):
                parser = CalldataFloat64Parser(key, chunk_length=chunk_length,
                                               n_samples=n_samples,
                                               number=n,
                                               fill=NAN)
            elif t.kind == 'S':
                parser = CalldataStringParser(key, dtype=t, chunk_length=chunk_length,
                                              n_samples=n_samples, number=n)
            # TODO unsigned int parsers
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for FORMAT field %r, field will be '
                              'skipped' % (t, key))
            context.calldata_parsers.append(parser)
        # TODO pointers

        #
        #
        # for key in formats:
        #     # TODO handle numbers
        #     # TODO handle types
        #     # TODO build a list of parsers and an array for nogil processing later
        #     t = types[key]
        #     n = numbers[key]
        #     if key == b'GT' and t == np.dtype('int8'):
        #         parser = GenotypeInt8Parser(key, chunk_length=chunk_length,
        #                                     n_samples=n_samples,
        #                                     ploidy=ploidy,
        #                                     fill=-1)
        #     elif key == b'GT' and t == np.dtype('int16'):
        #         parser = GenotypeInt16Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      ploidy=ploidy,
        #                                      fill=-1)
        #     elif key == b'GT' and t == np.dtype('int32'):
        #         parser = GenotypeInt32Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      ploidy=ploidy,
        #                                      fill=-1)
        #     elif key == b'GT' and t == np.dtype('int64'):
        #         parser = GenotypeInt64Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      ploidy=ploidy,
        #                                      fill=-1)
        #     elif t == np.dtype('int8'):
        #         parser = CalldataInt8Parser(key, chunk_length=chunk_length,
        #                                     n_samples=n_samples,
        #                                     number=n,
        #                                     fill=-1)
        #     elif t == np.dtype('int16'):
        #         parser = CalldataInt16Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      number=n,
        #                                      fill=-1)
        #     elif t == np.dtype('int32'):
        #         parser = CalldataInt32Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      number=n,
        #                                      fill=-1)
        #     elif t == np.dtype('int64'):
        #         parser = CalldataInt64Parser(key, chunk_length=chunk_length,
        #                                      n_samples=n_samples,
        #                                      number=n,
        #                                      fill=-1)
        #     elif t == np.dtype('float32'):
        #         parser = CalldataFloat32Parser(key, chunk_length=chunk_length,
        #                                        n_samples=n_samples,
        #                                        number=n,
        #                                        fill=NAN)
        #     elif t == np.dtype('float64'):
        #         parser = CalldataFloat64Parser(key, chunk_length=chunk_length,
        #                                        n_samples=n_samples,
        #                                        number=n,
        #                                        fill=NAN)
        #     elif t.kind == 'S':
        #         parser = CalldataStringParser(key, dtype=t, chunk_length=chunk_length,
        #                                       n_samples=n_samples, number=n)
        #     # TODO unsigned int parsers
        #     else:
        #         parser = self.skip_parser
        #         warnings.warn('type %s not supported for FORMAT field %r, field will be '
        #                       'skipped' % (t, key))
        #     self.parsers[key] = parser

    cdef void parse(self) nogil:
        CalldataParser_parse(self, self.context)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('CalldataParser.mkchunk')
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void CalldataParser_parse(CalldataParser self, ParserContext context) nogil:
    cdef:
        PyObject** parsers
        int i

    # initialise context
    context.sample_index = 0
    context.format_index = 0

    # initialise format parsers in correct order for this variant
    # TODO nogil version
    with gil:
        parsers = <PyObject **> malloc(context.n_formats * sizeof(PyObject*))
        for i, f in enumerate(context.formats):
            parser = <Parser> self.parsers.get(f, self.skip_parser)
            parsers[i] = <PyObject*> (<Parser> parser)
        # context.calldata_parsers = [self.parsers.get(f, self.skip_parser) for f in
        #                         context.formats]

    try:

        while True:

            if context.c == 0 or context.c == NEWLINE:
                context_getc(context)
                break

            elif context.c == TAB:

                context.sample_index += 1
                context.format_index = 0
                context_getc(context)

            elif context.c == COLON:

                context.format_index += 1
                context_getc(context)

            else:

                # check we haven't gone past last format parser
                if context.format_index < context.n_formats:
                    (<Parser>parsers[context.format_index]).parse(context)

    finally:
        free(parsers)


cdef class SkipInfoFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef void parse(self, ParserContext context) nogil:
        while context.c != SEMICOLON and context.c != TAB and context.c != 0:
            context_getc(context)


cdef class SkipCalldataFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext context) nogil:
        while context.c != COLON and context.c != TAB and context.c != NEWLINE and context.c != 0:
            context_getc(context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef inline void calldata_integer_parse(integer[:, :, :] memory,
                                        int number,
                                        ParserContext context) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            calldata_integer_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            calldata_integer_store(memory, number, context, value_index)
            break

        else:
            temp_append(context)

        context_getc(context)


cdef inline void calldata_integer_store(integer[:, :, :] memory,
                                        int number,
                                        ParserContext context,
                                        int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, value_index] = context.l


cdef inline void calldata_floating_parse(floating[:, :, :] memory,
                                         int number,
                                         ParserContext context) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            calldata_floating_store(memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            calldata_floating_store(memory, number, context, value_index)
            break

        else:
            temp_append(context)

        context_getc(context)


cdef inline void calldata_floating_store(floating[:, :, :] memory,
                                         int number,
                                         ParserContext context,
                                         int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as floating
    success = temp_todouble(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, value_index] = context.d


cdef class CalldataParserBase(Parser):

    cdef char* key
    cdef int number
    cdef int n_samples

    cdef void parse(self, ParserContext context) nogil:
        pass

    cdef void malloc(self):
        pass

    cdef void mkchunk(self, chunk, limit=None):
        # debug('CalldataParserBase.mkchunk')
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values
        self.malloc()


cdef class GenotypeInt8Parser(CalldataParserBase):

    cdef int ploidy
    cdef np.int8_t[:, :, :] memory
    cdef np.int8_t fill

    def __cinit__(self, bytes key, chunk_length, n_samples, ploidy, fill):
        self.key = PyBytes_AS_STRING(key)
        self.number = 1
        self.chunk_length = chunk_length
        self.n_samples = n_samples
        self.ploidy = ploidy
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.ploidy),
                               dtype='int8')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self, ParserContext context) nogil:
        genotype_parse(self.memory, context, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt8Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt16Parser(CalldataParserBase):

    cdef int ploidy
    cdef np.int16_t[:, :, :] memory
    cdef np.int16_t fill

    def __cinit__(self, bytes key, chunk_length, n_samples, ploidy, fill):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.n_samples = n_samples
        self.ploidy = ploidy
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.ploidy),
                               dtype='int16')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self, ParserContext context) nogil:
        genotype_parse(self.memory, context, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt8Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt32Parser(CalldataParserBase):

    cdef int ploidy
    cdef np.int32_t[:, :, :] memory
    cdef np.int32_t fill

    def __cinit__(self, bytes key, chunk_length, n_samples, ploidy, fill):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.n_samples = n_samples
        self.ploidy = ploidy
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.ploidy),
                               dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self, ParserContext context) nogil:
        genotype_parse(self.memory, context, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt32Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt64Parser(CalldataParserBase):

    cdef int ploidy
    cdef np.int64_t[:, :, :] memory
    cdef np.int64_t fill

    def __cinit__(self, bytes key, chunk_length, n_samples, ploidy, fill):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.n_samples = n_samples
        self.ploidy = ploidy
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.ploidy),
                               dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self, ParserContext context) nogil:
        genotype_parse(self.memory, context, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt64Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef inline int genotype_parse(integer[:, :, :] memory,
                               ParserContext context,
                               int ploidy) nogil except -1:
    cdef:
        int allele_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == SLASH or context.c == PIPE:
            genotype_store(memory, context, allele_index, ploidy)
            allele_index += 1
            temp_clear(context)

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE:
            genotype_store(memory, context, allele_index, ploidy)
            break

        else:
            temp_append(context)

        context_getc(context)

    return 1


cdef inline int genotype_store(integer[:, :, :] memory, ParserContext context,
                               int allele_index, int ploidy) nogil except -1:
    cdef:
        int success

    if allele_index >= ploidy:
        # more alleles than we've made room for, ignore
        return 0

    # attempt to parse allele
    success = temp_tolong(context)

    # store value
    if success:
        memory[context.chunk_variant_index, context.sample_index, allele_index] = context.l

    return 1


cdef class CalldataInt8Parser(CalldataParserBase):

    cdef np.int8_t[:, :, :] memory
    cdef np.int8_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_integer_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int8')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt16Parser(CalldataParserBase):

    cdef np.int16_t[:, :, :] memory
    cdef np.int16_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_integer_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int16')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt32Parser(CalldataParserBase):

    cdef np.int32_t[:, :, :] memory
    cdef np.int32_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_integer_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt64Parser(CalldataParserBase):

    cdef np.int64_t[:, :, :] memory
    cdef np.int64_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_integer_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill


# TODO unsigned int calldata parsers


cdef class CalldataFloat32Parser(CalldataParserBase):

    cdef np.float32_t[:, :, :] memory
    cdef np.float32_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_floating_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataFloat64Parser(CalldataParserBase):

    cdef np.float64_t[:, :, :] memory
    cdef np.float64_t fill

    def __cinit__(self, bytes key, fill, chunk_length, n_samples, number):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext context) nogil:
        calldata_floating_parse(self.memory, self.number, context)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataStringParser(CalldataParserBase):

    cdef object dtype
    cdef int itemsize
    cdef np.uint8_t[:] memory

    def __cinit__(self, key, chunk_length, dtype, number, n_samples):
        self.key = PyBytes_AS_STRING(key)
        self.chunk_length = chunk_length
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number
        self.n_samples = n_samples
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros((self.chunk_length, self.n_samples, self.number),
                               dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    cdef void parse(self, ParserContext context) nogil:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = ((context.chunk_variant_index *
                         self.n_samples *
                         self.number *
                         self.itemsize) +
                         (context.sample_index *
                          self.number *
                          self.itemsize))
        memory_index = memory_offset

        # read characters until tab
        while True:
            if context.c == TAB or context.c == COLON or context.c == NEWLINE or \
                    context.c == 0:
                break
            elif context.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0
            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = context.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1
            # advance input stream
            context_getc(context)

    cdef void mkchunk(self, chunk, limit=None):
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values
        self.malloc()
