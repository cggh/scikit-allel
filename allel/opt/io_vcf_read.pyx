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


cdef extern from "Python.h":
    char* PyByteArray_AS_STRING(object string)


ctypedef fused int_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


ctypedef fused float_t:
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


def iter_vcf(binary_file, int buffer_size, int chunk_length, int temp_max_size, headers, fields,
             types, numbers):
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

    # setup reader
    reader = BufferedReader(binary_file, buffer_size=buffer_size)

    # setup context
    n_samples = len(headers.samples)
    context = ParserContext(reader, temp_max_size=temp_max_size, n_samples=n_samples)

    # read in first character
    ParserContext_next(context)

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


cdef class BufferedReader(object):

    cdef object binary_file
    cdef int buffer_size
    cdef bytearray buffer
    cdef char* stream
    cdef char* stream_end

    def __cinit__(self, binary_file, buffer_size):
        self.binary_file = binary_file
        self.buffer_size = buffer_size
        self.buffer = bytearray(buffer_size)
        BufferedReader_fill_buffer(self)


# break out method as function for profiling
cdef inline void BufferedReader_fill_buffer(BufferedReader self):
    cdef:
        int l
    l = self.binary_file.readinto(self.buffer)
    # debug(l, self.buffer)
    if l > 0:
        self.stream = PyByteArray_AS_STRING(self.buffer)
        self.stream_end = self.stream + l
    else:
        self.stream = NULL


# break out method as function for profiling
cdef inline char BufferedReader_read(BufferedReader self):
    cdef:
        char c
    if self.stream == self.stream_end:
        BufferedReader_fill_buffer(self)
    if self.stream == NULL:
        return 0
    c = self.stream[0]
    self.stream += 1
    return c


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


cdef class ParserContext(object):

    cdef int state
    cdef bytes temp_bytes
    cdef char* temp
    cdef int temp_max_size
    cdef int temp_size
    cdef char c
    cdef BufferedReader reader
    cdef int n_samples
    cdef int variant_index
    cdef int chunk_variant_index
    cdef int sample_index
    cdef list formats
    cdef int format_index

    def __cinit__(self, reader, temp_max_size, n_samples):
        self.reader = reader
        self.state = ParserState.CHROM
        self.temp_bytes = PyBytes_FromStringAndSize(NULL, temp_max_size)
        self.temp = PyBytes_AS_STRING(self.temp_bytes)
        self.temp_max_size = temp_max_size
        self.temp_size = 0
        self.n_samples = n_samples
        self.variant_index = 0
        self.chunk_variant_index = 0
        self.sample_index = 0
        self.format_index = 0


cdef inline void temp_clear(ParserContext self):
    self.temp_size = 0


cdef inline void temp_append(ParserContext self):

    if self.temp_size < self.temp_max_size:

        # store current character
        self.temp[self.temp_size] = self.c

        # increase size
        self.temp_size += 1

    else:

        # TODO extend temporary buffer?
        raise RuntimeError('temporary buffer exceeded')


cdef inline long temp_strtol(ParserContext self, long default):
    cdef:
        char* str_end
        long value
        int parsed

    if self.temp_size == 0:

        warnings.warn('empty value at variant index %s' % self.variant_index)
        return default

    if self.temp_size == 1 and self.temp[0] == PERIOD:

        # explicit missing value
        return default

    if self.temp_size >= self.temp_max_size:

        # TODO extend temporary buffer?
        raise RuntimeError('temporary buffer exceeded')

    # terminate string
    self.temp[self.temp_size] = 0

    # do parsing
    value = strtol(self.temp, &str_end, 10)

    # check success
    parsed = str_end - self.temp

    # check success
    if self.temp_size == parsed:

        return value

    else:
        # TODO CHROM and POS in warning
        b = PyBytes_FromStringAndSize(self.temp, self.temp_size)

        if parsed > 0:
            warnings.warn('not all characters parsed for integer value %r from %r at '
                          'variant index %s' % (value, b, self.variant_index))
            return value

        else:
            warnings.warn('error %s parsing integer value %r at variant index %s' %
                          (value, b, self.variant_index))
            return default


cdef inline double temp_strtod(ParserContext self, double default):
    cdef:
        char* str_end
        double value
        int parsed

    if self.temp_size == 0:

        warnings.warn('empty value at variant index %s' % self.variant_index)
        return default

    if self.temp_size == 1 and self.temp[0] == PERIOD:

        # explicit missing value
        return default

    if self.temp_size >= self.temp_max_size:

        # TODO extend temporary buffer?
        raise RuntimeError('temporary buffer exceeded')

    # terminate string
    self.temp[self.temp_size] = 0

    # do parsing
    value = strtod(self.temp, &str_end)

    # check success
    parsed = str_end - self.temp

    # check success
    if self.temp_size == parsed:

        return value

    else:
        # TODO CHROM and POS in warning
        b = PyBytes_FromStringAndSize(self.temp, self.temp_size)

        if parsed > 0:
            warnings.warn('not all characters parsed for float value %r from %r at '
                          'variant index %s' % (value, b, self.variant_index))
            return value

        else:
            warnings.warn('error %s parsing float value %r at variant index %s' %
                          (value, b, self.variant_index))
            return default


cdef inline void ParserContext_next(ParserContext self):
    self.c = BufferedReader_read(self.reader)


cdef class Parser(object):
    """Abstract base class."""

    cdef object values
    cdef int chunk_length

    cdef parse(self, ParserContext context):
        pass

    cdef malloc(self):
        pass

    cdef mkchunk(self, chunk, limit=None):
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

    cdef malloc(self):
        self.values = np.zeros(self.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')

    cdef parse(self, ParserContext context):
        StringParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        chunk[self.field] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void StringParser_parse(StringParser self, ParserContext context):
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
        ParserContext_next(context)

    # advance input stream beyond tab
    ParserContext_next(context)

    # debug(context.variant_index, self.field, self.values[context.chunk_variant_index],
    #       chars_stored)


cdef class SkipChromParser(Parser):
    """Skip the CHROM field."""

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        SkipChromParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipChromParser_parse(SkipChromParser self, ParserContext context):

    # TODO store chrom on context

    # read characters until tab
    while context.c != TAB:
        ParserContext_next(context)

    # advance input stream beyond tab
    ParserContext_next(context)


cdef class PosInt32Parser(Parser):
    """Parser for POS field."""

    cdef np.int32_t[:] memory

    def __cinit__(self, chunk_length):
        self.chunk_length = chunk_length
        self.malloc()

    cdef malloc(self):
        self.values = np.zeros(self.chunk_length, dtype='i4')
        self.memory = self.values

    cdef parse(self, ParserContext context):
        PosInt32Parser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        chunk[POS_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void PosInt32Parser_parse(PosInt32Parser self, ParserContext context):
    cdef:
        long value

    # reset temporary buffer
    temp_clear(context)

    # read into temporary buffer until tab
    while context.c != TAB:
        temp_append(context)
        ParserContext_next(context)

    # parse string as integer
    value = temp_strtol(context, -1)

    # store value
    self.memory[context.chunk_variant_index] = value

    # advance input stream
    ParserContext_next(context)


cdef class SkipPosParser(Parser):
    """Skip the POS field."""

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        SkipPosParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        pass


# break out method as function for profiling
cdef inline void SkipPosParser_parse(SkipPosParser self, ParserContext context):

    # TODO store pos on context

    # read characters until tab
    while context.c != TAB:
        ParserContext_next(context)

    # advance input stream beyond tab
    ParserContext_next(context)


cdef class SkipFieldParser(Parser):
    """Skip a field."""

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        # read characters until tab or newline
        while context.c != TAB and context.c != NEWLINE and context.c != 0:
            ParserContext_next(context)

        # advance input stream beyond tab or newline
        ParserContext_next(context)

    cdef mkchunk(self, chunk, limit=None):
        pass


cdef class SkipAllCalldataParser(Parser):
    """Skip a field."""

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        # read characters until newline
        while context.c != NEWLINE and context.c != 0:
            ParserContext_next(context)

        # advance input stream beyond newline
        ParserContext_next(context)

    cdef mkchunk(self, chunk, limit=None):
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

    cdef malloc(self):
        self.values = np.zeros((self.chunk_length, self.number), dtype=self.dtype, order='C')
        self.memory = self.values.reshape(-1).view('u1')

    cdef parse(self, ParserContext context):
        AltParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[ALT_FIELD] = values
        self.malloc()


# break out method as function for profiling
cdef inline void AltParser_parse(AltParser self, ParserContext context):
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
            ParserContext_next(context)
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
        ParserContext_next(context)

    # debug(context.variant_index, 'ALT', self.values[context.chunk_variant_index])


cdef class QualFloat32Parser(Parser):

    cdef np.float32_t fill
    cdef np.float32_t[:] memory

    def __cinit__(self, chunk_length, fill):
        # debug('QualFloat32Parser __cinit__()')
        self.chunk_length = chunk_length
        self.fill = fill
        self.malloc()

    cdef malloc(self):
        # debug('QualFloat32Parser malloc()')
        self.values = np.empty(self.chunk_length, dtype='f4')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef parse(self, ParserContext context):
        QualFloat32Parser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        chunk[QUAL_FIELD] = self.values[:limit]
        self.malloc()


# break out method as function for profiling
cdef inline void QualFloat32Parser_parse(QualFloat32Parser self, ParserContext context):
    cdef:
        double value

    # reset temporary buffer
    temp_clear(context)

    # read into temporary buffer until tab
    while context.c != TAB:
        temp_append(context)
        ParserContext_next(context)

    # parse string as float
    value = temp_strtod(context, -1)

    # store value
    self.memory[context.chunk_variant_index] = value

    # advance input stream
    ParserContext_next(context)


cdef class FilterParser(Parser):

    cdef tuple filters
    cdef dict filter_position
    cdef np.uint8_t[:, :] memory

    def __cinit__(self, chunk_length, filters):
        self.chunk_length = chunk_length
        self.filters = tuple(filters)
        self.filter_position = {f: i for i, f in enumerate(self.filters)}
        self.malloc()

    cdef malloc(self):
        self.values = np.zeros((self.chunk_length, len(self.filters) + 1), dtype=bool)
        self.memory = self.values.view('u1')

    cdef parse(self, ParserContext context):
        FilterParser_parse(self, context)

    cdef store(self, ParserContext context):
        FilterParser_store(self, context)

    cdef mkchunk(self, chunk, limit=None):
        for i, filter in enumerate(self.filters):
            field = 'variants/FILTER_' + str(filter, 'ascii')
            # TODO any need to make it a contiguous array?
            chunk[field] = self.values[:limit, i]
        self.malloc()


# break out method as function for profiling
cdef inline void FilterParser_parse(FilterParser self, ParserContext context):
    cdef:
        int filter_index

    # reset temporary buffer
    temp_clear(context)

    # check for explicit missing value
    if context.c == PERIOD:
        while context.c != TAB:
            ParserContext_next(context)
        ParserContext_next(context)
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
        ParserContext_next(context)

    # advance to next field
    ParserContext_next(context)

    # debug(context.variant_index, 'FILTER', self.values[context.chunk_variant_index])


cdef inline void FilterParser_store(FilterParser self, ParserContext context):
    cdef:
        bytes f
        int filter_index

    if context.temp_size == 0:
        warnings.warn('found empty FILTER at variant index %s' %
                       context.variant_index)
        return

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
            elif t == np.dtype('i4'):
                parser = InfoInt32Parser(key, fill=-1, chunk_length=chunk_length,
                                              number=n)
            elif t == np.dtype('i8'):
                parser = InfoInt64Parser(key, fill=-1, chunk_length=chunk_length,
                                              number=n)
            elif t == np.dtype('f4'):
                parser = InfoFloat32Parser(key, fill=NAN,
                                                chunk_length=chunk_length,
                                                number=n)
            elif t == np.dtype('f8'):
                parser = InfoFloat64Parser(key, fill=NAN,
                                                chunk_length=chunk_length,
                                                number=n)
            elif t == np.dtype(bool):
                parser = InfoFlagParser(key, chunk_length=chunk_length)
            elif t.kind == 'S':
                parser = InfoStringMultiParser(key, chunk_length=chunk_length,
                                               dtype=t, number=n)
            else:
                parser = self.skip_parser
                warnings.warn('type %s not supported for INFO field %r, field will be skipped' %
                              (t, key))
            self.parsers[key] = parser

    cdef parse(self, ParserContext context):
        # debug(context.variant_index, 'InfoParser.parse')
        InfoParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void InfoParser_parse(InfoParser self, ParserContext context):
    cdef:
        bytes key
        Parser parser

    # debug(context.variant_index, 'InfoParser_parse')

    # check for explicit missing value
    if context.c == PERIOD:
        while context.c != TAB:
            ParserContext_next(context)
        ParserContext_next(context)
        return

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == TAB or context.c == NEWLINE or context.c == 0:
            # debug(context.variant_index, 'end of INFO')
            # handle flags
            if context.temp_size > 0:
                key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                parser = self.parsers.get(key, self.skip_parser)
                parser.parse(context)
            break

        elif context.c == EQUALS:
            # debug(context.variant_index, 'INFO =')
            ParserContext_next(context)
            if context.temp_size > 0:
                key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                # debug(context.variant_index, 'INFO parsing value for key', key)
                parser = self.parsers.get(key, self.skip_parser)
                parser.parse(context)
                temp_clear(context)
            else:
                warnings.warn('error parsing INFO field at variants index %s: missing key' %
                              (context.variant_index,))
                # advance to next sub-field
                while context.c != TAB and context.c != SEMICOLON and context.c != 0:
                    ParserContext_next(context)

        elif context.c == SEMICOLON:
            # debug(context.variant_index, 'end of INFO subfield')
            # handle flags
            if context.temp_size > 0:
                key = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                parser = self.parsers.get(key, self.skip_parser)
                # debug(context.variant_index, 'detected flag', key, parser)
                parser.parse(context)
                temp_clear(context)
            ParserContext_next(context)

        else:

            # debug(context.variant_index, 'storing INFO key character', bytes([context.c]))
            temp_append(context)
            ParserContext_next(context)

    # advance to next field
    ParserContext_next(context)


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

    cdef parse(self, ParserContext context):
        info_integer_parse(self.key, self.memory, self.number, context)

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='i4')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef mkchunk(self, chunk, limit=None):
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

    cdef parse(self, ParserContext context):
        info_integer_parse(self.key, self.memory, self.number, context)

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='i8')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef inline void info_integer_parse(bytes key, int_t[:, :] memory, int number,
                                          ParserContext context):
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:

            info_integer_store(key, memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == SEMICOLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            info_integer_store(key, memory, number, context, value_index)
            break

        else:

            temp_append(context)

        ParserContext_next(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void info_integer_store(bytes key, int_t[:, :] memory, int number,
                                          ParserContext context, int value_index):
    cdef:
        long value

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    # TODO configurable fill
    value = temp_strtol(context, -1)

    # store value
    memory[context.chunk_variant_index, value_index] = value


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

    cdef parse(self, ParserContext context):
        info_float_parse(self.key, self.memory, self.number, context)

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='f4')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef mkchunk(self, chunk, limit=None):
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

    cdef parse(self, ParserContext context):
        info_float_parse(self.key, self.memory, self.number, context)

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.number), dtype='f8')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef inline void info_float_parse(bytes key, float_t[:, :] memory, int number,
                                  ParserContext context):
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            info_float_store(key, memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == SEMICOLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            info_float_store(key, memory, number, context, value_index)
            break

        else:
            temp_append(context)

        ParserContext_next(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void info_float_store(bytes key, float_t[:, :] memory, int number,
                                  ParserContext context, int value_index):
    cdef:
        double value

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as double
    # TODO fill parameter
    value = temp_strtod(context, NAN)

    # store value
    memory[context.chunk_variant_index, value_index] = value


cdef class InfoFlagParser(Parser):

    cdef np.uint8_t[:] memory
    cdef bytes key

    def __cinit__(self, key, chunk_length):
        self.key = key
        self.chunk_length = chunk_length
        self.malloc()

    cdef parse(self, ParserContext context):
        self.memory[context.chunk_variant_index] = 1
        # ensure we advance the end of the field
        while context.c != SEMICOLON and context.c != TAB and context.c != NEWLINE and \
                context.c != 0:
            ParserContext_next(context)

    cdef malloc(self):
        self.values = np.zeros(self.chunk_length, dtype='u1')
        self.memory = self.values

    cdef mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        chunk[field] = self.values[:limit].view(bool)
        self.malloc()


cdef class InfoStringMultiParser(Parser):

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

    cdef malloc(self):
        self.values = np.zeros((self.chunk_length, self.number), dtype=self.dtype)
        self.memory = self.values.reshape(-1).view('u1')

    cdef parse(self, ParserContext context):
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        # read characters until tab
        while True:
            if context.c == TAB or context.c == SEMICOLON:
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
            ParserContext_next(context)

    cdef mkchunk(self, chunk, limit=None):
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values
        self.malloc()


cdef class FormatParser(Parser):

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        FormatParser_parse(self, context)


# break out method as function for profiling
cdef inline void FormatParser_parse(FormatParser self, ParserContext context):
    cdef:
        char* format
        bytes f
        list formats = []

    # debug('FormatParser_parse()')

    # reset temporary buffer
    temp_clear(context)

    while True:
        # debug(context.c)

        if context.c == TAB or context.c == NEWLINE:

            # add last format
            if context.temp_size > 0:
                # debug('add last format')
                f = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                formats.append(f)

            # set context
            context.formats = formats

            # we're done here
            break

        elif context.c == COLON:

            # add format
            if context.temp_size > 0:
                # debug('add format')
                f = PyBytes_FromStringAndSize(context.temp, context.temp_size)
                formats.append(f)
                temp_clear(context)

        else:
            # debug('other')

            temp_append(context)

        # advance to next character
        ParserContext_next(context)

    # advance to next field
    ParserContext_next(context)

    # debug(context.variant_index, 'FORMAT', formats)

# noinspection PyShadowingBuiltins
cdef class CalldataParser(Parser):

    cdef dict parsers
    cdef Parser skip_parser

    def __cinit__(self, chunk_length, formats, types, numbers, n_samples, ploidy):
        self.parsers = dict()
        for f in formats:
            # TODO handle numbers
            # TODO handle types
            if f == b'GT':
                self.parsers[f] = GenotypeInt8Parser(chunk_length=chunk_length,
                                                     n_samples=n_samples,
                                                     ploidy=ploidy,
                                                     fill=-1)
            # TODO initialise parsers for all fields
        self.skip_parser = SkipCalldataFieldParser()

    cdef parse(self, ParserContext context):
        CalldataParser_parse(self, context)

    cdef mkchunk(self, chunk, limit=None):
        cdef Parser parser
        for parser in self.parsers.values():
            parser.mkchunk(chunk, limit=limit)


# break out method as function for profiling
cdef inline void CalldataParser_parse(CalldataParser self, ParserContext context):
    cdef:
        list parsers
        Parser parser

    # initialise context
    context.sample_index = 0
    context.format_index = 0

    # initialise format parsers in correct order for this variant
    parsers = [self.parsers.get(f, self.skip_parser) for f in context.formats]
    parser = <Parser> parsers[0]

    while True:

        if context.c == 0 or context.c == NEWLINE:
            ParserContext_next(context)
            break

        elif context.c == TAB:

            context.sample_index += 1
            context.format_index = 0
            parser = <Parser> parsers[0]
            ParserContext_next(context)

        elif context.c == COLON:

            context.format_index += 1
            parser = <Parser> parsers[context.format_index]
            ParserContext_next(context)

        else:

            parser.parse(context)


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

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.ploidy), dtype='i1')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef parse(self, ParserContext context):
        GenotypeInt8Parser_parse(self, context)

    cdef store(self, ParserContext context, int allele_index):
        GenotypeInt8Parser_store(self, context, allele_index)

    cdef mkchunk(self, chunk, limit=None):
        chunk['calldata/GT'] = self.values[:limit]
        self.malloc()


cdef inline void GenotypeInt8Parser_parse(GenotypeInt8Parser self, ParserContext context):
    cdef:
        int allele_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == SLASH or context.c == PIPE:
            GenotypeInt8Parser_store(self, context, allele_index)
            allele_index += 1
            temp_clear(context)

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE:
            GenotypeInt8Parser_store(self, context, allele_index)
            break

        else:
            temp_append(context)

        ParserContext_next(context)


cdef inline void GenotypeInt8Parser_store(GenotypeInt8Parser self, ParserContext context,
                                          int allele_index):
    cdef:
        long allele

    if allele_index >= self.ploidy:
        # more alleles than we've made room for, ignore
        return

    # attempt to parse allele
    # TODO configurable missing
    allele = temp_strtol(context, -1)

    # store value
    self.memory[context.chunk_variant_index, context.sample_index, allele_index] = allele


cdef class SkipInfoFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef parse(self, ParserContext context):
        while context.c != SEMICOLON and context.c != TAB and context.c != 0:
            ParserContext_next(context)


cdef class SkipCalldataFieldParser(Parser):

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        while context.c != COLON and context.c != TAB and context.c != NEWLINE and context.c != 0:
            ParserContext_next(context)

    cdef mkchunk(self, chunk, limit=None):
        pass


cdef inline void calldata_integer_parse(bytes key, int_t[:, :, :] memory, int number,
                                        ParserContext context):
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            calldata_integer_store(key, memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            calldata_integer_store(key, memory, number, context, value_index)
            break

        else:
            temp_append(context)

        ParserContext_next(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void calldata_integer_store(bytes key, int_t[:, :, :] memory, int number,
                                        ParserContext context, int value_index):
    cdef:
        long value

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as integer
    # TODO configurable fill
    value = temp_strtol(context, -1)

    # store value
    memory[context.chunk_variant_index, context.sample_index, value_index] = value


cdef inline void calldata_float_parse(bytes key, float_t[:, :, :] memory, int number,
                                      ParserContext context):
    cdef:
        int value_index = 0

    # reset temporary buffer
    temp_clear(context)

    while True:

        if context.c == COMMA:
            calldata_float_store(key, memory, number, context, value_index)
            temp_clear(context)
            value_index += 1

        elif context.c == COLON or context.c == TAB or context.c == NEWLINE or \
                context.c == 0:
            calldata_float_store(key, memory, number, context, value_index)
            break

        else:
            temp_append(context)

        ParserContext_next(context)

    # reset temporary buffer here to indicate new field
    temp_clear(context)


cdef inline void calldata_float_store(bytes key, float_t[:, :, :] memory, int number,
                                      ParserContext context, int value_index):
    cdef:
        double value

    if value_index >= number:
        # more values than we have room for, ignore
        return

    # parse string as float
    # TODO configurable fill
    value = temp_strtod(context, NAN)

    # store value
    memory[context.chunk_variant_index, context.sample_index, value_index] = value


cdef class CalldataParserBase(Parser):

    cdef bytes key
    cdef int number
    cdef int n_samples

    cdef parse(self, ParserContext context):
        pass

    cdef malloc(self):
        pass

    cdef void mkchunk(self, chunk, limit=None):
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

    cdef parse(self, ParserContext context):
        calldata_integer_parse(self.key, self.memory, self.number, context)

    cdef malloc(self):
        self.values = np.empty((self.chunk_length, self.n_samples, self.number),
                               dtype='i1')
        self.memory = self.values
        self.memory[:] = self.fill
