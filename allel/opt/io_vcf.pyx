# cython: language_level=3
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
"""


import sys
import gzip
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stdlib cimport strtol, strtof
import numpy as np
cimport numpy as np
import cython
cimport cython


cdef char TAB = b'\t'
cdef char NEWLINE = b'\n'
cdef char HASH = b'#'
cdef char COLON = b':'
cdef char PERIOD = b'.'
cdef char COMMA = b','
cdef char SLASH = b'/'
cdef char PIPE = b'|'


def debug(*msg):
    print(*msg, file=sys.stderr)
    sys.stderr.flush()


cdef class BufferedInputStream(object):

    cdef object fileobj
    cdef int buffer_size
    cdef bytes buffer
    cdef char* stream
    cdef char* stream_end

    def __cinit__(self, fileobj, buffer_size):
        self.fileobj = fileobj
        self.buffer_size = buffer_size
        BufferedInputStream_fill_buffer(self)


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.profile(False)
cpdef inline void BufferedInputStream_fill_buffer(BufferedInputStream self):
    cdef:
        int l
    self.buffer = self.fileobj.read(self.buffer_size)
    l = len(self.buffer)
    if l > 0:
        self.stream = PyBytes_AS_STRING(self.buffer)
        self.stream_end = self.stream + l
    else:
        self.stream = NULL


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.profile(False)
cpdef inline char BufferedInputStream_next(BufferedInputStream self):
    cdef:
        char c
    if self.stream == self.stream_end:
        BufferedInputStream_fill_buffer(self)
    if self.stream == NULL:
        return 0
    c = self.stream[0]
    self.stream += 1
    return c


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def spike_read_len(fn, buffer_size):
    """Foo."""
    cdef:
        BufferedInputStream input_stream
        char c
        int i = 0
    with gzip.open(fn, mode='rb') as fileobj:
        input_stream = BufferedInputStream(fileobj, buffer_size=buffer_size)
        c = BufferedInputStream_next(input_stream)
        while c != 0:
            c = BufferedInputStream_next(input_stream)
            i += 1
    return i


def foo():
    pass


def spike_read(fn, buffer_size, limit):
    cdef:
        BufferedInputStream input_stream
        char c
        int i = 0
    with gzip.open(fn, mode='rb') as fileobj:
        input_stream = BufferedInputStream(fileobj, buffer_size=buffer_size)
        c = BufferedInputStream_next(input_stream)
        while c != 0 and i < limit:
            debug(i, <bytes>c)
            c = BufferedInputStream_next(input_stream)
            i += 1


cdef enum ParserState:
    HEADER,
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
    cdef char* input_stream
    cdef int n_samples
    cdef int variant_index
    cdef int block_variant_index
    cdef int sample_index
    cdef list formats
    cdef int format_index
    cdef char* header_start

    def __cinit__(self):
        self.state = ParserState.HEADER
        self.n_samples = 0
        self.variant_index = 0
        self.block_variant_index = 0
        self.sample_index = 0
        self.format_index = 0
        self.header_start = NULL


cdef class Parser(object):
    """Abstract base class."""

    cdef parse(self, ParserContext context):
        pass

    cdef malloc(self):
        pass


def check_string_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != 'S':
        raise ValueError('expected byte string ("S") dtype, found: %r' % dtype)
    return dtype


cdef class HeaderParser(Parser):

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        HeaderParser_parse(self, context)


cpdef HeaderParser_parse(HeaderParser self, ParserContext context):
    cdef:
        char c = context.input_stream[0]

    debug('HeaderParser_parse', <int>(context.input_stream), c)

    # if context.header_start == NULL:
    #
    #     if c == HASH:
    #         context.header_start = context.input_stream
    #
    #
    # if c != HASH:
    #
    #     # TODO review this
    #     raise RuntimeError('missing final header line?')
    #
    # while c != NEWLINE:
    #     context.input_stream += 1
    #     c = context.input_stream[0]
    #
    # header_len = context.input_stream - header_start
    # debug(header_len)
    # header_line = PyBytes_FromStringAndSize(header_start, header_len)
    # debug(header_line)
    #
    # if header_line.startswith(b'#CHROM'):
    #
    #     # record number of samples
    #     context.n_samples = len(header_line.split(b'\t')) - 9
    #
    #     # advance state
    #     context.state = ParserState.CHROM
    #
    # context.input_stream += 1


cdef class StringParser(Parser):
    """Generic string column parser, used for CHROM, ID, REF."""

    cdef int block_size
    cdef object dtype
    cdef int itemsize
    cdef object values
    cdef np.uint8_t[:] memory

    def __cinit__(self, block_size, dtype):
        self.block_size = block_size
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.malloc()

    cdef malloc(self):
        self.values = np.zeros(self.block_size, dtype=self.dtype)
        self.memory = self.values.view('u1')

    cdef parse(self, ParserContext context):
        StringParser_parse(self, context)


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef StringParser_parse(StringParser self, ParserContext context):
    cdef:
        # current character in input buffer
        char c = context.input_stream[0]
        # index into memory view
        int memory_index
        # number of characters read into current value
        int chars_stored = 0

    # initialise memory index
    memory_index = context.block_variant_index * self.itemsize

    # read characters until tab
    while c != TAB:
        if chars_stored < self.itemsize:
            # store value
            self.memory[memory_index] = c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context.input_stream += 1
        # read next character
        c = context.input_stream[0]

    # advance input stream beyond tab
    context.input_stream += 1


cdef class PosInt32Parser(Parser):
    """Parser for POS field."""

    cdef int block_size
    cdef object values
    cdef np.int32_t[:] memory

    def __cinit__(self, block_size):
        self.block_size = block_size

    cdef malloc(self):
        self.values = np.zeros(self.block_size, dtype='i4')
        self.memory = self.values

    cdef parse(self, ParserContext context):
        PosInt32Parser_parse(self, context)


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef PosInt32Parser_parse(PosInt32Parser self, ParserContext context):
    cdef:
        long value
        char* str_end

    # parse string as integer
    value = strtol(context.input_stream, &str_end, 10)

    # check success
    if str_end > context.input_stream:

        # store value
        self.memory[context.block_variant_index] = value

        # advance input stream
        context.input_stream = str_end + 1

    else:
        raise RuntimeError('error %s parsing POS at variant index %s' %
                           (value, context.variant_index))


cdef class AltParser(Parser):
    """Parser for ALT field."""

    cdef int block_size
    cdef object dtype
    cdef int itemsize
    cdef int arity
    cdef object values
    cdef np.uint8_t[:] memory

    def __cinit__(self, block_size, dtype, arity):
        self.block_size = block_size
        self.dtype = check_string_dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.arity = arity

    cdef malloc(self):
        self.values = np.zeros((self.block_size, self.arity),
                               dtype=self.dtype, order='C')
        self.memory = self.values.view('u1')

    cdef parse(self, ParserContext context):
        AltParser_parse(self, context)


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef AltParser_parse(AltParser self, ParserContext context):
    cdef:
        # current character in input buffer
        char c = context.input_stream[0]
        # index of alt values
        int alt_index = 0
        # index into memory view
        int memory_offset, memory_index
        # number of characters read into current value
        int chars_stored = 0

    # initialise memory offset and index
    memory_offset = context.block_variant_index * self.itemsize * self.arity
    memory_index = memory_offset

    # read characters until tab
    while True:
        if c == TAB:
            context.input_stream += 1
            break
        elif c == COMMA:
            # advance value index
            alt_index += 1
            # set memory index to beginning of next item
            memory_index = memory_offset + (alt_index * self.itemsize)
            # reset chars stored
            chars_stored = 0
        elif chars_stored < self.itemsize:
            # store value
            self.memory[memory_index] = c
            # advance memory index
            memory_index += 1
            # advance number of characters stored
            chars_stored += 1
        # advance input stream
        context.input_stream += 1
        # read next character
        c = context.input_stream[0]


cdef class QualFloat32Parser(Parser):

    cdef int block_size
    cdef np.float32_t fill
    cdef object values
    cdef np.float32_t[:] memory

    def ___cinit___(self, block_size, fill):
        self.block_size = block_size
        self.fill = fill

    cdef malloc(self):
        self.values = np.empty(self.block_size, dtype='f4')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef parse(self, ParserContext context):
        QualFloat32Parser_parse(self, context)


# break out method as function for profiling
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef QualFloat32Parser_parse(QualFloat32Parser self,
                                    ParserContext context):
    cdef:
        float value
        char* str_end

    # parse string as float
    value = strtof(context.input_stream, &str_end)

    # check success
    if str_end > context.input_stream:

        # store value
        self.memory[context.block_variant_index] = value

        # advance input stream
        context.input_stream = str_end

    elif context.input_stream[0] == TAB:
        # handle completely missing value - not strictly kosher

        # advance input stream to next column
        context.input_stream += 1

    elif context.input_stream[0] == PERIOD and context.input_stream[1] == TAB:
        # handle explicit missing value

        # advance input stream to next column
        context.input_stream += 2

    else:
        raise RuntimeError('error %s parsing QUAL at variant index %s' %
                           (value, context.variant_index))


cdef class FilterParser(Parser):

    cdef int block_size
    cdef tuple filters
    cdef dict filter_position
    cdef object values
    cdef np.uint8_t[:, :] memory

    def __cinit__(self, block_size, filters):
        self.block_size = block_size
        self.filters = tuple(filters)
        # PASS comes first
        self.filter_position = {f: i + 1 for i, f in enumerate(self.filters)}
        self.filter_position[b'PASS'] = 0

    cdef malloc(self):
        self.values = np.zeros((self.block_size, len(self.filters) + 1), dtype=bool)
        self.memory = self.values.view('u1')

    cdef parse(self, ParserContext context):
        FilterParser_parse(self, context)

    cdef store(self, ParserContext context, char* filter_start, int filter_len):
        FilterParser_store(self, context, filter_start, filter_len)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FilterParser_store(FilterParser self,
                         ParserContext context,
                         char* filter_start,
                         int filter_len):
        # TODO needs optimising?
        cdef:
            bytes f
            int filter_index

        # read filter into byte string
        f = PyBytes_FromStringAndSize(filter_start, filter_len)

        # find filter position
        try:
            filter_index = self.filter_position[f]
        except KeyError as e:
            raise RuntimeError('unexpected FILTER: %s' % str(f, 'ascii'))

        # store value
        self.memory[context.block_variant_index, filter_index] = 1


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FilterParser_parse(FilterParser self, ParserContext context):
    cdef:
        char c = context.input_stream[0]
        char* filter_start = context.input_stream
        int filter_index

    if c == TAB:
        # handle completely missing value - not strictly kosher

        # advance input stream to next column
        context.input_stream += 1

    elif c == PERIOD and context.input_stream[1] == TAB:
        # handle explicit missing value

        # advance input stream to next column
        context.input_stream += 2

    else:
        # parse filters

        while True:

            if c == COMMA:  # TODO semi-colon, colon?

                # store true
                self.store(context, filter_start, context.input_stream - filter_start)

                # advance input stream
                context.input_stream += 1
                filter_start = context.input_stream
                c = context.input_stream[0]

            elif c == TAB:

                # store true
                self.store(context, filter_start, context.input_stream - filter_start)

                # advance input stream
                context.input_stream += 1
                break

            else:

                # advance input stream
                context.input_stream += 1
                c = context.input_stream[0]


cdef class InfoParser(Parser):

    def __cinit__(self, block_size):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        InfoParser_parse(self, context)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef InfoParser_parse(InfoParser self, ParserContext context):
    # TODO
    cdef:
        char c = context.input_stream[0]

    while c != TAB:
        context.input_stream += 1
        c = context.input_stream[0]

    context.input_stream += 1


cdef class FormatParser(Parser):

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        FormatParser_parse(self, context)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FormatParser_parse(FormatParser self, ParserContext context):
    cdef:
        char c = context.input_stream[0]
        char* format_start = context.input_stream
        list formats = []
        bytes fmt

    while True:

        if c == TAB:

            # add last format
            fmt = PyBytes_FromStringAndSize(format_start,
                                            context.input_stream - format_start)
            formats.append(fmt)

            # set context
            context.formats = formats

            # advance and break
            context.input_stream += 1
            break

        elif c == COLON:

            # add format
            fmt = PyBytes_FromStringAndSize(format_start,
                                            context.input_stream - format_start)
            formats.append(fmt)
            context.input_stream += 1
            format_start = context.input_stream
            c = context.input_stream[0]

        else:

            # advance
            context.input_stream += 1
            c = context.input_stream[0]


cdef class CalldataParser(Parser):

    cdef dict parsers

    def __cinit__(self, block_size, formats, n_samples, ploidy):
        self.parsers = dict()
        for f in formats:
            if f == b'GT':
                self.parsers[f] = GenotypeInt8Parser(block_size=block_size,
                                                     n_samples=n_samples,
                                                     ploidy=ploidy,
                                                     fill=-1)
            else:
                self.parsers[f] = DummyCalldataParser()
            # TODO initialise parsers for all fields

    cdef malloc(self):
        for parser in self.parsers.values():
            parser.malloc()

    cdef parse(self, ParserContext context):
        CalldataParser_parse(self, context)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CalldataParser_parse(CalldataParser self, ParserContext context):
    cdef:
        list parsers
        Parser parser
        char c = context.input_stream[0]

    # initialise context
    context.sample_index = 0
    context.format_index = 0

    # initialise format parsers in correct order for this variant
    parsers = [self.parsers[f] for f in context.formats]
    parser = <Parser> parsers[0]

    while True:

        if c == NEWLINE:

            context.input_stream += 1
            break

        elif c == TAB:

            context.input_stream += 1
            context.sample_index += 1
            context.format_index = 0
            parser = <Parser> parsers[0]

        elif c == COLON:

            context.input_stream += 1
            context.format_index += 1
            parser = <Parser> parsers[context.format_index]

        else:

            parser.parse(context)

        c = context.input_stream[0]


cdef class GenotypeInt8Parser(Parser):

    cdef int block_size
    cdef int n_samples
    cdef int ploidy
    cdef object values
    cdef np.int8_t[:, :, :] memory
    cdef np.int8_t fill

    def __cinit__(self, block_size, n_samples, ploidy, fill):
        self.block_size = block_size
        self.n_samples = n_samples
        self.ploidy = ploidy
        self.fill = fill

    cdef malloc(self):
        self.values = np.empty((self.block_size, self.n_samples, self.ploidy), dtype='i1')
        self.memory = self.values
        self.memory[:] = self.fill

    cdef parser(self, ParserContext context):
        GenotypeInt8Parser_parse(self, context)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef GenotypeInt8Parser_parse(GenotypeInt8Parser self, ParserContext context):
    cdef:
        int allele_index = 0
        long allele
        char* str_end
        char c = context.input_stream[0]

    while True:

        if c == PERIOD:
            context.input_stream += 1
        elif c == SLASH or c == PIPE:
            allele_index += 1
            context.input_stream += 1
        elif c == COLON or c == TAB:
            break
        elif allele_index < self.ploidy:

            # parse allele as integer
            allele = strtol(context.input_stream, &str_end, 10)

            # store value
            if str_end > context.input_stream:
                self.memory[context.block_variant_index, context.sample_index, allele_index] = \
                    allele
                context.input_stream = str_end

            else:
                raise RuntimeError('error %s parsing genotype at variant index %s, sample index '
                                   '%s' % (allele, context.variant_index, context.sample_index))
        else:

            # we are beyond ploidy
            context.input_stream += 1

        c = context.input_stream[0]


cdef class DummyCalldataParser(Parser):

    def __cinit__(self):
        pass

    cdef malloc(self):
        pass

    cdef parser(self, ParserContext context):
        DummyCalldataParser_parse(self, context)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DummyCalldataParser_parse(DummyCalldataParser self, ParserContext context):
    cdef:
        char c = context.input_stream[0]

    while True:

        if c == COLON or c == TAB:
            break

        else:
            context.input_stream += 1

        c = context.input_stream[0]


def vcf_block_iter(fileobj, buffer_size, block_size):
    cdef:
        ParserContext context
        char* stream_end
        Parser header_parser
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

    # setup
    context = ParserContext()
    # TODO user-provided dtypes
    header_parser = HeaderParser()
    chrom_parser = StringParser(block_size=block_size, dtype='S12')
    pos_parser = PosInt32Parser(block_size=block_size)
    id_parser = StringParser(block_size=block_size, dtype='S12')
    ref_parser = StringParser(block_size=block_size, dtype='S12')
    alt_parser = AltParser(block_size=block_size, dtype='S1', arity=3)
    qual_parser = QualFloat32Parser(block_size=block_size)
    format_parser = FormatParser()

    # TODO yield blocks
    blocks = []

    # read in first buffer
    buffer = fileobj.read(buffer_size)

    while buffer:

        # obtain buffer as C string
        context.input_stream = PyBytes_AS_STRING(buffer)

        # end of buffer
        stream_end = context.input_stream + len(buffer)

        # iterate character by character until end of buffer
        while context.input_stream < stream_end:

            if context.state == ParserState.HEADER:
                header_parser.parse(context)

                # detect transition from header to body
                if context.state == ParserState.CHROM:
                    # TODO discover filters from header
                    filter_parser = FilterParser(block_size=block_size, filters=[])
                    # TODO discuver INFO fields from header
                    info_parser = InfoParser(block_size=block_size)
                    # TODO handle all format fields
                    calldata_parser = CalldataParser(block_size=block_size,
                                                     formats=[b'GT'],
                                                     n_samples=context.n_samples,
                                                     ploidy=2)

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
                info_parser.parse(context)
                context.state = ParserState.FORMAT

            elif context.state == ParserState.FORMAT:
                format_parser.parse(context)
                context.state = ParserState.CALLDATA

            elif context.state == ParserState.CALLDATA:
                calldata_parser.parse(context)
                context.state = ParserState.CHROM

                context.variant_index += 1
                if context.block_variant_index < block_size:
                    context.block_variant_index += 1
                else:
                    context.block_variant_index = 0
                    block = {
                        'variants/CHROM': chrom_parser.values,
                        'variants/POS': pos_parser.values,
                        'variants/REF': ref_parser.values,
                        'variants/ALT': alt_parser.values,
                        'variants/QUAL': qual_parser.values,
                        'variants/FILTER': filter_parser.values,
                        # TODO INFO
                        'calldata/GT': calldata_parser.parsers['GT'].values,
                        # TODO other calldata
                    }
                    blocks.append(block)

            else:
                raise Exception('unexpected parser state')

        # read in next buffer
        buffer = fileobj.read(buffer_size)

    # left-over block
    l = context.block_variant_index + 1
    block = {
        'variants/CHROM': chrom_parser.values[:l],
        'variants/POS': pos_parser.values[:l],
        'variants/REF': ref_parser.values[:l],
        'variants/ALT': alt_parser.values[:l],
        'variants/QUAL': qual_parser.values[:l],
        'variants/FILTER': filter_parser.values[:l],
        # TODO INFO
        'calldata/GT': calldata_parser.parsers['GT'].values[:l],
        # TODO other calldata
    }
    blocks.append(block)

    return blocks


def vcf_block_read(path, buffer_size, block_size):

    if isinstance(path, str) and path.endswith('gz'):
        with gzip.open(path, mode='rb') as fileobj:
            return vcf_block_iter(fileobj, buffer_size=buffer_size, block_size=block_size)

    elif isinstance(path, str):
        with open(path, mode='rb') as fileobj:
            return vcf_block_iter(fileobj, buffer_size=buffer_size, block_size=block_size)

    else:
        return vcf_block_iter(path, buffer_size=buffer_size, block_size=block_size)
