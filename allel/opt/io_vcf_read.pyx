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
from libc.stdlib cimport strtol, strtof, strtod
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


cdef void warn(message, ParserContext ctx) nogil:
    with gil:
        message += '; variant index: %s' % ctx.variant_index
        b = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
        message += '; temporary buffer: %s' % b
        warnings.warn(message)


def iter_vcf(input_file, int input_buffer_size, int chunk_length, int temp_buffer_size,
             headers, fields, types, numbers):
    cdef:
        ParserContext ctx
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
    ctx = ParserContext(input_file=input_file,
                        input_buffer_size=input_buffer_size,
                        temp_buffer_size=temp_buffer_size,
                        n_samples=n_samples)

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
                                         n_samples=ctx.n_samples,
                                         ploidy=2)
    else:
        format_parser = SkipFieldParser()
        calldata_parser = SkipAllCalldataParser()

    if fields:
        # shouldn't ever be any left over
        raise RuntimeError('unexpected fields left over: %r' % set(fields))

    while True:

        if ctx.c == 0:
            break

        elif ctx.state == ParserState.CHROM:
            chrom_parser.parse(ctx)
            ctx.state = ParserState.POS

        elif ctx.state == ParserState.POS:
            pos_parser.parse(ctx)
            ctx.state = ParserState.ID

        elif ctx.state == ParserState.ID:
            id_parser.parse(ctx)
            ctx.state = ParserState.REF

        elif ctx.state == ParserState.REF:
            ref_parser.parse(ctx)
            ctx.state = ParserState.ALT

        elif ctx.state == ParserState.ALT:
            alt_parser.parse(ctx)
            ctx.state = ParserState.QUAL

        elif ctx.state == ParserState.QUAL:
            qual_parser.parse(ctx)
            ctx.state = ParserState.FILTER

        elif ctx.state == ParserState.FILTER:
            filter_parser.parse(ctx)
            ctx.state = ParserState.INFO

        elif ctx.state == ParserState.INFO:
            # debug(ctx.variant_index, 'parse INFO')
            info_parser.parse(ctx)
            ctx.state = ParserState.FORMAT

        elif ctx.state == ParserState.FORMAT:
            format_parser.parse(ctx)
            ctx.state = ParserState.CALLDATA

        elif ctx.state == ParserState.CALLDATA:
            calldata_parser.parse(ctx)
            ctx.state = ParserState.CHROM

            # setup next variant
            # debug('setup next variant')
            ctx.variant_index += 1
            if ctx.chunk_variant_index < chunk_length - 1:
                ctx.chunk_variant_index += 1

            else:

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
                ctx.chunk_variant_index = 0

        else:
            # shouldn't ever happen
            raise RuntimeError('unexpected parser state')

    # left-over chunk
    limit = ctx.chunk_variant_index
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
        object input_file
        int input_buffer_size
        bytearray input_buffer
        char* input
        char* input_start
        char* input_end
        int state
        int temp_buffer_size
        bytearray temp_buffer
        char* temp
        int temp_size
        char c
        long l
        double d
        int n_samples
        int variant_index
        int chunk_variant_index
        int sample_index
        list formats
        int format_index
        list calldata_parsers


    def __cinit__(self,
                  input_file,
                  int input_buffer_size,
                  int temp_buffer_size,
                  int n_samples):

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


cdef inline int context_fill_buffer(ParserContext ctx) nogil except -1:
    cdef:
        int l
    with gil:
        l = ctx.input_file.readinto(ctx.input_buffer)
    if l > 0:
        ctx.input = ctx.input_start
        ctx.input_end = ctx.input + l
        return 1
    else:
        ctx.input = NULL
        return 0


cdef inline int context_getc(ParserContext ctx) nogil except -1:

    if ctx.input is ctx.input_end:
        context_fill_buffer(ctx)

    if ctx.input is NULL:
        ctx.c = 0
        return 0

    else:
        ctx.c = ctx.input[0]
        ctx.input += 1
        return 1


cdef inline void temp_clear(ParserContext ctx) nogil:
    ctx.temp_size = 0


cdef inline int temp_append(ParserContext ctx) nogil except -1:

    # if ctx.temp_size >= ctx.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # store current character
    ctx.temp[ctx.temp_size] = ctx.c

    # increase size
    ctx.temp_size += 1

    return 1


cdef inline int temp_tolong(ParserContext ctx) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if ctx.temp_size == 0:

        warn('expected integer, found empty value', ctx)
        return 0

    if ctx.temp_size == 1 and ctx.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # if ctx.temp_size >= ctx.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # terminate string
    ctx.temp[ctx.temp_size] = 0

    # do parsing
    ctx.l = strtol(ctx.temp, &str_end, 10)

    # check success
    parsed = str_end - ctx.temp

    # check success
    if ctx.temp_size == parsed:

        return 1

    else:

        if parsed > 0:
            warn('not all characters parsed for integer value', ctx)
            return 1

        else:
            warn('error parsing integer value', ctx)
            return 0


cdef inline int temp_todouble(ParserContext ctx) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if ctx.temp_size == 0:

        warn('expected floating, found empty value', ctx)
        return 0

    if ctx.temp_size == 1 and ctx.temp[0] == PERIOD:

        # explicit missing value
        return 0

    # if ctx.temp_size >= ctx.temp_buffer_size:
    #
    #     # TODO extend temporary buffer
    #     pass

    # terminate string
    ctx.temp[ctx.temp_size] = 0

    # do parsing
    ctx.d = strtod(ctx.temp, &str_end)

    # check success
    parsed = str_end - ctx.temp

    # check success
    if ctx.temp_size == parsed:

        return 1

    else:

        if parsed > 0:
            warn('not all characters parsed for floating value', ctx)
            return 1

        else:
            warn('error parsing floating value', ctx)
            return 0


cdef class Parser(object):
    """Abstract base class."""

    cdef object values
    cdef int chunk_length

    cdef void parse(self, ParserContext context) nogil:
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

    def __cinit__(self, field, chunk_length, dtype):
        self.field = field
        self.chunk_length = chunk_length
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros(self.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    cdef void parse(self, ParserContext context) nogil:
        StringParser_parse(self, context)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void StringParser_parse(StringParser self, ParserContext ctx) nogil:
    cdef:
        # index into memory view
        int memory_index
        # number of characters read into current value
        int chars_stored = 0

    # debug('StringParser_parse', self.field)

    # initialise memory index
    memory_index = ctx.chunk_variant_index * self.itemsize

    # read characters until tab
    while ctx.c != TAB:
        if chars_stored < self.itemsize:
            # store value
            self.memory[memory_index] = ctx.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context_getc(ctx)

    # advance input stream beyond tab
    context_getc(ctx)

    # debug(context.variant_index, self.field, self.values[context.chunk_variant_index],
    #       chars_stored)


cdef class SkipChromParser(Parser):
    """Skip the CHROM field."""

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext context) nogil:
        SkipChromParser_parse(self, context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipChromParser_parse(SkipChromParser self, ParserContext ctx) nogil:

    # TODO store chrom on ctx

    # read characters until tab
    while ctx.c != TAB:
        context_getc(ctx)

    # advance input stream beyond tab
    context_getc(ctx)


cdef class PosInt32Parser(Parser):
    """Parser for POS field."""

    cdef np.int32_t[:] memory

    def __cinit__(self, chunk_length):
        self.chunk_length = chunk_length
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros(self.chunk_length, dtype='int32')
        self.memory = self.values
        self.memory[:] = -1

    cdef void parse(self, ParserContext context) nogil:
        PosInt32Parser_parse(self, context)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[POS_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void PosInt32Parser_parse(PosInt32Parser self, ParserContext ctx) nogil:
    cdef:
        long value
        int success

    # reset temporary buffer
    temp_clear(ctx)

    # read into temporary buffer until tab
    while ctx.c != TAB:
        temp_append(ctx)
        context_getc(ctx)

    # parse string as integer
    success = temp_tolong(ctx)

    # store value
    if success:
        self.memory[ctx.chunk_variant_index] = ctx.l

    # advance input stream
    context_getc(ctx)


cdef class SkipPosParser(Parser):
    """Skip the POS field."""

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext context) nogil:
        SkipPosParser_parse(self, context)

    cdef void mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipPosParser_parse(SkipPosParser self, ParserContext ctx) nogil:

    # TODO store pos on ctx

    # read characters until tab
    while ctx.c != TAB:
        context_getc(ctx)

    # advance input stream beyond tab
    context_getc(ctx)


cdef class SkipFieldParser(Parser):
    """Skip a field."""

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext ctx) nogil:
        # read characters until tab or newline
        while ctx.c != TAB and ctx.c != NEWLINE and ctx.c != 0:
            context_getc(ctx)

        # advance input stream beyond tab or newline
        context_getc(ctx)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef class SkipAllCalldataParser(Parser):
    """Skip a field."""

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext ctx) nogil:
        # read characters until newline
        while ctx.c != NEWLINE and ctx.c != 0:
            context_getc(ctx)

        # advance input stream beyond newline
        context_getc(ctx)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef class AltParser(Parser):
    """Parser for ALT field."""

    cdef object dtype
    cdef int itemsize
    cdef int number
    cdef np.uint8_t[:] memory

    def __cinit__(self, chunk_length, dtype, number):
        self.chunk_length = chunk_length
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros((self.chunk_length, self.number), dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    cdef void parse(self, ParserContext ctx) nogil:
        AltParser_parse(self, ctx)

    cdef void mkchunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values
        self.malloc()


# break out method as function for profiling
cdef inline void AltParser_parse(AltParser self, ParserContext ctx) nogil:
    cdef:
        # index of alt values
        int alt_index = 0
        # index into memory view
        int memory_offset, memory_index
        # number of characters read into current value
        int chars_stored = 0

    # initialise memory offset and index
    memory_offset = ctx.chunk_variant_index * self.itemsize * self.number
    memory_index = memory_offset

    # read characters until tab
    while True:
        if ctx.c == TAB:
            context_getc(ctx)
            break
        elif ctx.c == COMMA:
            # advance value index
            alt_index += 1
            # set memory index to beginning of next item
            memory_index = memory_offset + (alt_index * self.itemsize)
            # reset chars stored
            chars_stored = 0
        elif chars_stored < self.itemsize and alt_index < self.number:
            # store value
            self.memory[memory_index] = ctx.c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context_getc(ctx)

    # debug(ctx.variant_index, 'ALT', self.values[ctx.chunk_variant_index])


cdef class QualFloat32Parser(Parser):

    cdef np.float32_t fill
    cdef np.float32_t[:] memory

    def __cinit__(self, chunk_length, fill):
        # debug('QualFloat32Parser __cinit__()')
        self.chunk_length = chunk_length
        self.fill = fill
        self.malloc()

    cdef void malloc(self):
        # debug('QualFloat32Parser malloc()')
        self.values = np.empty(self.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void parse(self, ParserContext ctx) nogil:
        QualFloat32Parser_parse(self, ctx)

    cdef void mkchunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void QualFloat32Parser_parse(QualFloat32Parser self, ParserContext ctx) nogil:
    cdef:
        int success

    # reset temporary buffer
    temp_clear(ctx)

    # read into temporary buffer until tab
    while ctx.c != TAB:
        temp_append(ctx)
        context_getc(ctx)

    # parse string as floating
    success = temp_todouble(ctx)

    # store value
    if success:
        self.memory[ctx.chunk_variant_index] = ctx.d

    # advance input stream
    context_getc(ctx)


cdef class FilterParser(Parser):

    cdef tuple filters
    cdef dict filter_position
    cdef np.uint8_t[:, :] memory

    def __cinit__(self, chunk_length, filters):
        self.chunk_length = chunk_length
        self.filters = tuple(filters)
        self.filter_position = {f: i for i, f in enumerate(self.filters)}
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros((self.chunk_length, len(self.filters) + 1), dtype=bool)
        self.memory = self.values.view('u1')

    cdef void parse(self, ParserContext ctx) nogil:
        FilterParser_parse(self, ctx)

    cdef store(self, ParserContext ctx):
        FilterParser_store(self, ctx)

    cdef void mkchunk(self, chunk, limit=None):
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]
        self.malloc()


# break out method as function for profiling
cdef inline void FilterParser_parse(FilterParser self, ParserContext ctx) nogil:
    cdef:
        int filter_index

    # reset temporary buffer
    temp_clear(ctx)

    # check for explicit missing value
    if ctx.c == PERIOD:
        while ctx.c != TAB:
            context_getc(ctx)
        context_getc(ctx)
        return

    while True:

        if ctx.c == TAB or ctx.c == NEWLINE or ctx.c == 0:
            FilterParser_store(self, ctx)
            break

        elif ctx.c == COMMA or ctx.c == COLON or ctx.c == SEMICOLON:
            # some of these delimiters are not strictly kosher, but have seen them
            FilterParser_store(self, ctx)
            temp_clear(ctx)

        else:
            temp_append(ctx)

        # advance to next character
        context_getc(ctx)

    # advance to next field
    context_getc(ctx)

    # debug(ctx.variant_index, 'FILTER', self.values[ctx.chunk_variant_index])


cdef inline void FilterParser_store(FilterParser self, ParserContext ctx) nogil:
    cdef:
        int filter_index

    if ctx.temp_size == 0:
        warn('empty FILTER', ctx)
        return

    # TODO nogil version?

    with gil:

        # read filter into byte string
        f = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)

        # find filter position
        filter_index = self.filter_position.get(f, -1)

    # store value
    if filter_index >= 0:
        self.memory[ctx.chunk_variant_index, filter_index] = 1

    # debug(ctx.variant_index, 'FILTER', f)


cdef class InfoParser(Parser):

    cdef tuple infos
    cdef dict parsers
    cdef Parser skip_parser

    def __cinit__(self, chunk_length, infos, types, numbers):
        self.chunk_length = chunk_length
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

    cdef void parse(self, ParserContext ctx) nogil:
        # debug(ctx.variant_index, 'InfoParser.parse')
        InfoParser_parse(self, ctx)

    cdef void mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void InfoParser_parse(InfoParser self, ParserContext ctx) nogil:
    # cdef:
    #     Parser parser

    # debug(ctx.variant_index, 'InfoParser_parse')

    # check for explicit missing value
    if ctx.c == PERIOD:
        while ctx.c != TAB:
            context_getc(ctx)
        context_getc(ctx)
        return

    # reset temporary buffer
    temp_clear(ctx)

    with gil:
        # TODO nogil version?

        while True:

            if ctx.c == TAB or ctx.c == NEWLINE or ctx.c == 0:
                # debug(ctx.variant_index, 'end of INFO')
                # handle flags
                if ctx.temp_size > 0:
                    key = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
                    parser = self.parsers.get(key, self.skip_parser)
                    parser.parse(ctx)
                break

            elif ctx.c == EQUALS:
                # debug(ctx.variant_index, 'INFO =')
                context_getc(ctx)
                if ctx.temp_size > 0:
                    key = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
                    # debug(ctx.variant_index, 'INFO parsing value for key', key)
                    parser = self.parsers.get(key, self.skip_parser)
                    parser.parse(ctx)
                    temp_clear(ctx)
                else:
                    warn('error parsing INFO field, missing key', ctx)
                    # advance to next sub-field
                    while ctx.c != TAB and ctx.c != SEMICOLON and ctx.c != 0:
                        context_getc(ctx)

            elif ctx.c == SEMICOLON:
                # debug(ctx.variant_index, 'end of INFO subfield')
                # handle flags
                if ctx.temp_size > 0:
                    key = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
                    (<Parser>self.parsers.get(key, self.skip_parser)).parse(ctx)
                    # debug(ctx.variant_index, 'detected flag', key, parser)
                    # parser.parse(ctx)
                    temp_clear(ctx)
                context_getc(ctx)

            else:

                # debug(ctx.variant_index, 'storing INFO key character', bytes([ctx.c]))
                temp_append(ctx)
                context_getc(ctx)

    # advance to next field
    context_getc(ctx)


cdef class InfoInt32Parser(Parser):

    cdef np.int32_t[:, :] memory
    cdef bytes key
    cdef object fill
    cdef int number

    def __cinit__(self, key, fill, chunk_length, number):
        self.key = key
        self.fill = fill
        self.chunk_length = chunk_length
        self.number = number
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        info_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class InfoInt64Parser(Parser):

    cdef np.int64_t[:, :] memory
    cdef np.int64_t fill
    cdef bytes key
    cdef int number

    def __cinit__(self, key, fill, chunk_length, number):
        self.key = key
        self.fill = fill
        self.chunk_length = chunk_length
        self.number = number
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        info_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef inline void info_integer_parse(bytes key, integer[:, :] memory, int number,
                                          ParserContext ctx) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(ctx)

    while True:

        if ctx.c == COMMA:

            info_integer_store(key, memory, number, ctx, value_index)
            temp_clear(ctx)
            value_index += 1

        elif ctx.c == SEMICOLON or ctx.c == TAB or ctx.c == NEWLINE or \
                ctx.c == 0:
            info_integer_store(key, memory, number, ctx, value_index)
            break

        else:

            temp_append(ctx)

        context_getc(ctx)

    # reset temporary buffer here to indicate new field
    temp_clear(ctx)


cdef inline void info_integer_store(bytes key, integer[:, :] memory, int number,
                                    ParserContext ctx, int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    success = temp_tolong(ctx)

    # store value
    if success:
        memory[ctx.chunk_variant_index, value_index] = ctx.l


cdef class InfoFloat32Parser(Parser):

    cdef np.float32_t[:, :] memory
    cdef np.float32_t fill
    cdef bytes key
    cdef int number

    def __cinit__(self, key, fill, chunk_length, number):
        self.key = key
        self.fill = fill
        self.number = number
        self.chunk_length = chunk_length
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        info_floating_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class InfoFloat64Parser(Parser):

    cdef np.float64_t[:, :] memory
    cdef np.float64_t fill
    cdef bytes key
    cdef int number

    def __cinit__(self, key, fill, chunk_length, number):
        self.key = key
        self.fill = fill
        self.number = number
        self.chunk_length = chunk_length
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        info_floating_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef inline void info_floating_parse(bytes key, floating[:, :] memory, int number,
                                  ParserContext ctx) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(ctx)

    while True:

        if ctx.c == COMMA:
            info_floating_store(key, memory, number, ctx, value_index)
            temp_clear(ctx)
            value_index += 1

        elif ctx.c == SEMICOLON or ctx.c == TAB or ctx.c == NEWLINE or \
                ctx.c == 0:
            info_floating_store(key, memory, number, ctx, value_index)
            break

        else:
            temp_append(ctx)

        context_getc(ctx)

    # reset temporary buffer here to indicate new field
    temp_clear(ctx)


cdef inline void info_floating_store(bytes key, floating[:, :] memory, int number,
                                  ParserContext ctx, int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as double
    success = temp_todouble(ctx)

    # store value
    if success:
        memory[ctx.chunk_variant_index, value_index] = ctx.d


cdef class InfoFlagParser(Parser):

    cdef np.uint8_t[:] memory
    cdef bytes key

    def __cinit__(self, key, chunk_length):
        self.key = key
        self.chunk_length = chunk_length
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        self.memory[ctx.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while ctx.c != SEMICOLON and ctx.c != TAB and ctx.c != NEWLINE and \
                ctx.c != 0:
            context_getc(ctx)

    cdef void malloc(self):
        self.values = np.zeros(self.chunk_length, dtype='u1')
        self.memory = self.values

    cdef void mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)
        self.malloc()


cdef class InfoStringParser(Parser):

    cdef bytes key
    cdef object dtype
    cdef int itemsize
    cdef np.uint8_t[:] memory
    cdef int number

    def __cinit__(self, key, chunk_length, dtype, number):
        self.key = key
        self.chunk_length = chunk_length
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.number = number
        self.malloc()

    cdef void malloc(self):
        self.values = np.zeros((self.chunk_length, self.number), dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    cdef void parse(self, ParserContext ctx) nogil:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = ctx.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        # read characters until tab
        while True:
            if ctx.c == TAB or ctx.c == SEMICOLON:
                break
            elif ctx.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0
            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = ctx.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1
            # advance input stream
            context_getc(ctx)

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

    cdef void parse(self, ParserContext ctx) nogil:
        FormatParser_parse(self, ctx)


# break out method as function for profiling
cdef inline void FormatParser_parse(FormatParser self, ParserContext ctx) nogil:
    cdef:
        char* format

    # debug('FormatParser_parse()')

    # reset temporary buffer
    temp_clear(ctx)

    with gil:

        ctx.formats = []

        while True:
            # debug(ctx.c)

            if ctx.c == TAB or ctx.c == NEWLINE:

                # add last format
                if ctx.temp_size > 0:
                    # debug('add last format')
                    f = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
                    ctx.formats.append(f)

                # we're done here
                break

            elif ctx.c == COLON:

                # add format
                if ctx.temp_size > 0:
                    # debug('add format')
                    f = PyBytes_FromStringAndSize(ctx.temp, ctx.temp_size)
                    ctx.formats.append(f)
                    temp_clear(ctx)

            else:
                # debug('other')

                temp_append(ctx)

            # advance to next character
            context_getc(ctx)

    # advance to next field
    context_getc(ctx)

    # debug(ctx.variant_index, 'FORMAT', formats)

# noinspection PyShadowingBuiltins
cdef class CalldataParser(Parser):

    cdef tuple formats
    cdef dict parsers
    cdef Parser skip_parser

    def __cinit__(self, chunk_length, formats, types, numbers, n_samples, ploidy):
        self.chunk_length = chunk_length
        self.formats = tuple(formats)
        self.parsers = dict()
        self.skip_parser = SkipCalldataFieldParser()
        for key in formats:
            # TODO handle numbers
            # TODO handle types
            t = types[key]
            n = numbers[key]
            if key == b'GT' and t == np.dtype('int8'):
                parser = GenotypeInt8Parser(chunk_length=chunk_length,
                                            n_samples=n_samples,
                                            ploidy=ploidy,
                                            fill=-1)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = GenotypeInt16Parser(chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             ploidy=ploidy,
                                             fill=-1)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = GenotypeInt32Parser(chunk_length=chunk_length,
                                             n_samples=n_samples,
                                             ploidy=ploidy,
                                             fill=-1)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = GenotypeInt64Parser(chunk_length=chunk_length,
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
            self.parsers[key] = parser

    cdef void parse(self, ParserContext ctx) nogil:
        CalldataParser_parse(self, ctx)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('CalldataParser.mkchunk')
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void CalldataParser_parse(CalldataParser self, ParserContext ctx) nogil:

    # initialise ctx
    ctx.sample_index = 0
    ctx.format_index = 0

    # initialise format parsers in correct order for this variant
    with gil:
        ctx.calldata_parsers = [self.parsers.get(f, self.skip_parser) for f in
                                ctx.formats]

    with nogil:

        while True:

            if ctx.c == 0 or ctx.c == NEWLINE:
                context_getc(ctx)
                break

            elif ctx.c == TAB:

                ctx.sample_index += 1
                ctx.format_index = 0
                context_getc(ctx)

            elif ctx.c == COLON:

                ctx.format_index += 1
                context_getc(ctx)

            else:

                # TODO check we haven't gone past last format parser
                (<Parser>PyList_GET_ITEM(ctx.calldata_parsers,
                                         ctx.format_index)).parse(ctx)
                # parser = <Parser> parsers[ctx.format_index]
                # parser = <Parser> PyList_GET_ITEM(parsers, ctx.format_index)
                # parser.parse(ctx)


cdef class GenotypeInt8Parser(Parser):

    cdef int n_samples
    cdef int ploidy
    cdef np.int8_t[:, :, :] memory
    cdef np.int8_t fill

    def __cinit__(self, chunk_length, n_samples, ploidy, fill):
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

    cdef void parse(self, ParserContext ctx) nogil:
        genotype_parse(self.memory, ctx, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt8Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt16Parser(Parser):

    cdef int n_samples
    cdef int ploidy
    cdef np.int16_t[:, :, :] memory
    cdef np.int16_t fill

    def __cinit__(self, chunk_length, n_samples, ploidy, fill):
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

    cdef void parse(self, ParserContext ctx) nogil:
        genotype_parse(self.memory, ctx, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt8Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt32Parser(Parser):

    cdef int n_samples
    cdef int ploidy
    cdef np.int32_t[:, :, :] memory
    cdef np.int32_t fill

    def __cinit__(self, chunk_length, n_samples, ploidy, fill):
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

    cdef void parse(self, ParserContext ctx) nogil:
        genotype_parse(self.memory, ctx, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt32Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef class GenotypeInt64Parser(Parser):

    cdef int n_samples
    cdef int ploidy
    cdef np.int64_t[:, :, :] memory
    cdef np.int64_t fill

    def __cinit__(self, chunk_length, n_samples, ploidy, fill):
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

    cdef void parse(self, ParserContext ctx) nogil:
        genotype_parse(self.memory, ctx, self.ploidy)

    cdef void mkchunk(self, chunk, limit=None):
        # debug('GenotypeInt64Parser.mkchunk')
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef inline int genotype_parse(integer[:, :, :] memory,
                               ParserContext ctx,
                               int ploidy) nogil except -1:
    cdef:
        int allele_index = 0

    # reset temporary buffer
    temp_clear(ctx)

    while True:

        if ctx.c == SLASH or ctx.c == PIPE:
            genotype_store(memory, ctx, allele_index, ploidy)
            allele_index += 1
            temp_clear(ctx)

        elif ctx.c == COLON or ctx.c == TAB or ctx.c == NEWLINE:
            genotype_store(memory, ctx, allele_index, ploidy)
            break

        else:
            temp_append(ctx)

        context_getc(ctx)

    return 1


cdef inline int genotype_store(integer[:, :, :] memory, ParserContext ctx,
                               int allele_index, int ploidy) nogil except -1:
    cdef:
        int success

    if allele_index >= ploidy:
        # more alleles than we've made room for, ignore
        return 0

    # attempt to parse allele
    success = temp_tolong(ctx)

    # store value
    if success:
        memory[ctx.chunk_variant_index, ctx.sample_index, allele_index] = ctx.l

    return 1


cdef class SkipInfoFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef void parse(self, ParserContext ctx) nogil:
        while ctx.c != SEMICOLON and ctx.c != TAB and ctx.c != 0:
            context_getc(ctx)


cdef class SkipCalldataFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef void malloc(self):
        pass

    cdef void parse(self, ParserContext ctx) nogil:
        while ctx.c != COLON and ctx.c != TAB and ctx.c != NEWLINE and ctx.c != 0:
            context_getc(ctx)

    cdef void mkchunk(self, chunk, limit=None):
        pass


cdef inline void calldata_integer_parse(bytes key, integer[:, :, :] memory, int number,
                                        ParserContext ctx) nogil:
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(ctx)

    while True:

        if ctx.c == COMMA:
            calldata_integer_store(key, memory, number, ctx, value_index)
            temp_clear(ctx)
            value_index += 1

        elif ctx.c == COLON or ctx.c == TAB or ctx.c == NEWLINE or \
                ctx.c == 0:
            calldata_integer_store(key, memory, number, ctx, value_index)
            break

        else:
            temp_append(ctx)

        context_getc(ctx)


cdef inline void calldata_integer_store(bytes key, integer[:, :, :] memory, int number,
                                        ParserContext ctx, int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    success = temp_tolong(ctx)

    # store value
    if success:
        memory[ctx.chunk_variant_index, ctx.sample_index, value_index] = ctx.l


cdef inline void calldata_floating_parse(bytes key, floating[:, :, :] memory, int number,
                                      ParserContext ctx):
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(ctx)

    while True:

        if ctx.c == COMMA:
            calldata_floating_store(key, memory, number, ctx, value_index)
            temp_clear(ctx)
            value_index += 1

        elif ctx.c == COLON or ctx.c == TAB or ctx.c == NEWLINE or \
                ctx.c == 0:
            calldata_floating_store(key, memory, number, ctx, value_index)
            break

        else:
            temp_append(ctx)

        context_getc(ctx)


cdef inline void calldata_floating_store(bytes key, floating[:, :, :] memory, int number,
                                      ParserContext ctx, int value_index) nogil:
    cdef:
        int success

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as floating
    success = temp_todouble(ctx)

    # store value
    if success:
        memory[ctx.chunk_variant_index, ctx.sample_index, value_index] = ctx.d


cdef class CalldataParserBase(Parser):

    cdef bytes key
    cdef int number
    cdef int n_samples

    cdef void parse(self, ParserContext ctx) nogil:
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


cdef class CalldataInt8Parser(CalldataParserBase):

    cdef np.int8_t[:, :, :] memory
    cdef np.int8_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int8')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt16Parser(CalldataParserBase):

    cdef np.int16_t[:, :, :] memory
    cdef np.int16_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int16')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt32Parser(CalldataParserBase):

    cdef np.int32_t[:, :, :] memory
    cdef np.int32_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataInt64Parser(CalldataParserBase):

    cdef np.int64_t[:, :, :] memory
    cdef np.int64_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_integer_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='int64')
        self.memory = self.values
        self.memory[:] = self.fill


# TODO unsigned int calldata parsers


cdef class CalldataFloat32Parser(CalldataParserBase):

    cdef np.float32_t[:, :, :] memory
    cdef np.float32_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_floating_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='float32')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataFloat64Parser(CalldataParserBase):

    cdef np.float64_t[:, :, :] memory
    cdef np.float64_t fill

    def __cinit__(self, key, fill, chunk_length, n_samples, number):
        self.chunk_length = chunk_length
        self.key = key
        self.number = number
        self.n_samples = n_samples
        self.fill = fill
        self.malloc()

    cdef void parse(self, ParserContext ctx) nogil:
        calldata_floating_parse(self.key, self.memory, self.number, ctx)

    cdef void malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='float64')
        self.memory = self.values
        self.memory[:] = self.fill


cdef class CalldataStringParser(Parser):

    cdef bytes key
    cdef object dtype
    cdef int itemsize
    cdef np.uint8_t[:] memory
    cdef int number
    cdef int n_samples

    def __cinit__(self, key, chunk_length, dtype, number, n_samples):
        self.key = key
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

    cdef void parse(self, ParserContext ctx) nogil:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = ((ctx.chunk_variant_index *
                         self.n_samples *
                         self.number *
                         self.itemsize) +
                         (ctx.sample_index *
                          self.number *
                          self.itemsize))
        memory_index = memory_offset

        # read characters until tab
        while True:
            if ctx.c == TAB or ctx.c == COLON or ctx.c == NEWLINE or \
                    ctx.c == 0:
                break
            elif ctx.c == COMMA:
                # advance value index
                value_index += 1
                # set memory index to beginning of next item
                memory_index = memory_offset + (value_index * self.itemsize)
                # reset chars stored
                chars_stored = 0
            elif chars_stored < self.itemsize and value_index < self.number:
                # store value
                self.memory[memory_index] = ctx.c
                # advance memory index
                memory_index += 1
                # advance number of characters stored
                chars_stored += 1
            # advance input stream
            context_getc(ctx)

    cdef void mkchunk(self, chunk, limit=None):
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values
        self.malloc()
