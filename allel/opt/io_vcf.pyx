# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1


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


cdef enum ParserState:
    HEADER_NEWLINE,
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
    cdef int variant_index
    cdef int block_variant_index
    cdef int sample_index
    cdef list formats
    cdef int format_index

    def __cinit__(self):
        self.state = ParserState.HEADER_NEWLINE
        self.variant_index = 0
        self.block_variant_index = 0
        self.sample_index = 0
        self.format_index = 0


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
cdef AltParser_parse(AltParser self, ParserContext context):
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
        self.filters = filters
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


cpdef FilterParser_store(FilterParser self,
                               ParserContext context,
                               char* filter_start,
                               int filter_len):
        # TODO needs optimising?
        cdef:
            bytes f

        # read filter into byte string
        f = PyBytes_FromStringAndSize(filter_start, filter_len)

        # find filter position
        try:
            filter_index = self.filter_position[f]
        except KeyError as e:
            raise RuntimeError('unexpected FILTER: %s' % str(f, 'ascii'))

        # store value
        self.memory[context.block_variant_index, filter_index] = 1


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

    def __cinit__(self, size):
        pass

    cdef malloc(self):
        pass

    cdef parse(self, ParserContext context):
        InfoParser_parse(self, context)


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

    def __cinit__(self, formats):
        for f in formats:
            if f == 'GT':
                self.parsers[f] = GenotypeInt8Parser()
            else:
                self.parsers[f] = DummyCalldataParser()
            # TODO initialise parsers for all fields

    cdef malloc(self):
        for parser in self.parsers.values():
            parser.malloc()

    cdef parse(self, ParserContext context):
        CalldataParser_parse(self, context)


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
            context.variant_index += 1
            context.block_variant_index += 1
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
    chrom_parser = StringParser()

    # read in first buffer
    buffer = fileobj.read(buffer_size)

    while buffer:

        # obtain buffer as C string
        context.input_stream = PyBytes_AS_STRING(buffer)

        # end of buffer
        stream_end = context.input_stream + len(buffer)

        # iterate character by character until end of buffer
        while context.input_stream < stream_end:

            if context.state == ParserState.HEADER_NEWLINE:

                parse_header_newline(parser)

            elif parser.state == States.HEADER:

                parse_header(parser)

            elif parser.state == States.CHROM:

                parse_chrom(parser)

            elif parser.state == States.POS:

                parse_pos(parser)

            elif parser.state == States.ID:

                parse_id(parser)

            elif parser.state == States.REF:

                parse_ref(parser)

            elif parser.state == States.ALT:

                parse_alt(parser)

            elif parser.state == States.QUAL:

                parse_qual(parser)

            elif parser.state == States.FILTER:

                parse_filter(parser)

            elif parser.state == States.INFO:

                parse_info(parser)

            elif parser.state == States.FORMAT:

                parse_format(parser)

            elif parser.state == States.CALLDATA:

                parse_calldata(parser)

            else:
                raise Exception('unexpected parser state')

        # read in next buffer
        buffer = fileobj.read(buffer_size)
