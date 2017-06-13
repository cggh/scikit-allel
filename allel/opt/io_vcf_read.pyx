# cython: language_level=3
# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
"""
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: profile=True
# cython: binding=True
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


#########################################################################################
# Constants.


# for Windows compatibility
cdef double NAN = np.nan

# predefine useful characters
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

# useful to lookup max int values
II8 = np.iinfo(np.int8)
II16 = np.iinfo(np.int16)
II32 = np.iinfo(np.int32)
II64 = np.iinfo(np.int64)
IU8 = np.iinfo(np.uint8)
IU16 = np.iinfo(np.uint16)
IU32 = np.iinfo(np.uint32)
IU64 = np.iinfo(np.uint64)


##########################################################################################
# Fused Types.


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


##########################################################################################
# Vectors, written in pure C for speed and ease of use without GIL.


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
# C string utilities.


cdef inline int cstr_search_sorted(char* query, char** compare, int n_items) nogil:
    cdef:
        int i

    # N.B., we could do a binary search here, but in fact this is not the performance
    # bottleneck, so stick with a simple scan for now

    for i in range(n_items):
        if strcmp(query, compare[i]) == 0:
            return i

    return -1


##########################################################################################
# General I/O utilities.


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
# VCF Parsing.


cdef enum VCFState:
    CHROM = 0,
    POS = 1,
    ID = 2,
    REF = 3,
    ALT = 4,
    QUAL = 5,
    FILTER = 6,
    INFO = 7,
    FORMAT = 8,
    CALLDATA = 9,
    EOL = 10,
    EOF = 11


cdef class VCFContext:

    cdef:
        # useful stuff
        object headers

        # dynamic attributes - reflect current state during parsing
        int state  # overall parser state
        int variant_index  # index of current variant
        int chunk_variant_index  # index of current variant within current chunk
        int sample_index  # index of current sample within call data
        int sample_output_index  # index of current sample within output calldata arrays
        int sample_field_index  # index of field within call data for current sample
        IntVector variant_format_indices  # indices of formats for the current variant

        # buffers
        CharVector temp  # used for numeric values
        CharVector info_key  # used for info key
        CharVector info_val  # used for info value

        # keep track of current chrom and pos, even if fields are skipped
        CharVector chrom
        long pos

        # track size of reference allele (needed for svlen)
        int ref_len

    def __cinit__(self, headers):
        self.headers = headers

        # initialise dynamic state
        self.state = VCFState.CHROM
        self.variant_index = -1
        self.chunk_variant_index = -1
        self.sample_index = 0
        self.sample_output_index = -1
        self.sample_field_index = 0
        IntVector_init(&self.variant_format_indices, 2**6)

        # initialise temporary buffers
        CharVector_init(&self.temp, 2**6)
        CharVector_init(&self.info_key, 2**6)
        CharVector_init(&self.info_val, 2**6)

        # initialise chrom and pos
        CharVector_init(&self.chrom, 2**6)
        self.pos = -1
        self.ref_len = 0

    def __dealloc__(self):
        IntVector_free(&self.variant_format_indices)
        CharVector_free(&self.temp)
        CharVector_free(&self.info_key)
        CharVector_free(&self.info_val)
        CharVector_free(&self.chrom)


def check_samples(samples, headers):
    n_samples = len(headers.samples)
    if samples is None:
        samples = np.ones(n_samples, dtype='u1')
    else:
        # assume samples is already a boolean indexing array
        samples = samples.view('u1')
        assert samples.shape[0] == n_samples
    return samples


cdef class VCFChunkIterator:

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
                 numbers,
                 fills,
                 region,
                 samples):

        # store reference to input stream
        self.stream = stream

        # setup context
        self.context = VCFContext(headers)

        # setup parser
        samples = check_samples(samples, headers)
        self.parser = VCFParser(fields=fields, types=types, numbers=numbers,
                                chunk_length=chunk_length, samples=samples,
                                fills=fills, region=region)

    def __iter__(self):
        return self

    def __next__(self):

        if self.context.state == VCFState.EOF:
            raise StopIteration

        # reset indices
        self.context.chunk_variant_index = -1

        # allocate arrays for next chunk
        self.parser.malloc_chunk()

        # parse next chunk
        self.parser.parse(self.stream, self.context)

        # get the chunk
        chunk_length = self.context.chunk_variant_index + 1
        chunk = self.parser.make_chunk(chunk_length)

        if chunk is None:
            raise StopIteration

        chrom = CharVector_to_pybytes(&self.context.chrom)
        pos = self.context.pos
        return chunk, chunk_length, chrom, pos


cdef class VCFParser:

    cdef:
        int chunk_length
        np.uint8_t[:] samples
        VCFFieldParserBase chrom_pos_parser
        VCFFieldParserBase id_parser
        VCFFieldParserBase ref_parser
        VCFFieldParserBase alt_parser
        VCFFieldParserBase qual_parser
        VCFFieldParserBase filter_parser
        VCFFieldParserBase info_parser
        VCFFieldParserBase format_parser
        VCFFieldParserBase calldata_parser
        bytes region_chrom
        int region_begin
        int region_end

    def __init__(self, fields, types, numbers, chunk_length, samples, fills, region):
        self.chunk_length = chunk_length
        self.samples = samples

        # copy so we don't modify someone else's data
        fields = set(fields)

        # handle region
        self._init_region(region)

        # setup parsers
        self._init_chrom_pos(fields, types)
        self._init_id(fields, types)
        self._init_ref(fields, types)
        self._init_alt(fields, types, numbers)
        self._init_qual(fields, types, fills)
        self._init_filter(fields)
        self._init_info(fields, types, numbers, fills)
        self._init_format_calldata(fields, types, numbers, fills)

        if fields:
            # shouldn't ever be any left over
            raise RuntimeError('unexpected fields left over: %r' % set(fields))

    def _init_region(self, region):
        self.region_chrom = b''
        self.region_begin = 0
        self.region_end = 0
        if region is not None:
            tokens = region.split(':')
            if len(tokens) == 0:
                raise ValueError('bad region string: %r' % region)
            self.region_chrom = tokens[0].encode('ascii')
            if len(tokens) > 1:
                range_tokens = tokens[1].split('-')
                if len(range_tokens) != 2:
                    raise ValueError('bad region string: %r' % region)
                self.region_begin = int(range_tokens[0])
                self.region_end = int(range_tokens[1])

    def _init_chrom_pos(self, fields, types):
        """Setup CHROM and POS parser."""
        kwds = dict(dtype=None, chunk_length=self.chunk_length,
                    region_chrom=self.region_chrom, region_begin=self.region_begin,
                    region_end=self.region_end, store_chrom=False, store_pos=False)

        if CHROM_FIELD in fields:
            kwds['dtype'] = types[CHROM_FIELD]
            kwds['store_chrom'] = True
            fields.remove(CHROM_FIELD)

        if POS_FIELD in fields:
            if POS_FIELD in types:
                t = types[POS_FIELD]
                if t != np.dtype('int32'):
                    warnings.warn('only int32 supported for POS field, ignoring requested type: %r' % t)
            kwds['store_pos'] = True
            fields.remove(POS_FIELD)

        chrom_pos_parser = VCFChromPosParser(**kwds)
        chrom_pos_parser.malloc_chunk()
        self.chrom_pos_parser = chrom_pos_parser

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
        t = types.get(REF_FIELD, None)
        store = False
        if REF_FIELD in fields:
            store = True
            fields.remove(REF_FIELD)
        ref_parser = VCFRefParser(dtype=t, chunk_length=self.chunk_length, store=store)
        ref_parser.malloc_chunk()
        self.ref_parser = ref_parser

    def _init_alt(self, fields, types, numbers):
        """Setup ALT parser."""

        store_alt = False
        store_numalt = False
        store_svlen = False
        t = types.get(ALT_FIELD, None)
        n = numbers.get(ALT_FIELD, 1)
        if ALT_FIELD in fields:
            store_alt = True
            fields.remove(ALT_FIELD)
        if 'variants/numalt' in fields:
            store_numalt = True
            fields.remove('variants/numalt')
        if 'variants/svlen' in fields:
            store_svlen = True
            fields.remove('variants/svlen')

        if store_alt or store_numalt or store_svlen:
            alt_parser = VCFAltParser(dtype=t, number=n, chunk_length=self.chunk_length, store_alt=store_alt,
                                      store_numalt=store_numalt, store_svlen=store_svlen)
        else:
            alt_parser = VCFSkipFieldParser(key=b'ALT')

        alt_parser.malloc_chunk()
        self.alt_parser = alt_parser

    def _init_qual(self, fields, types, fills):
        """Setup QUAL parser."""
        if QUAL_FIELD in fields:
            if QUAL_FIELD in types:
                t = types[QUAL_FIELD]
                if t != np.dtype('float32'):
                    warnings.warn('only float32 supported for QUAL field, ignoring requested type: %r' % t)
            fill = fills.get(QUAL_FIELD, -1)
            qual_parser = VCFQualParser(chunk_length=self.chunk_length, fill=fill)
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
        if filter_keys:
            filter_parser = VCFFilterParser(filters=filter_keys,
                                            chunk_length=self.chunk_length)
        else:
            filter_parser = VCFSkipFieldParser(key=b'FILTER')
        filter_parser.malloc_chunk()
        self.filter_parser = filter_parser

    def _init_info(self, fields, types, numbers, fills):
        # setup INFO parser
        info_keys = list()
        info_types = dict()
        info_numbers = dict()
        info_fills = dict()
        # assume any variants fields left are INFO
        for field in list(fields):
            group, name = field.split('/')
            if group == 'variants':
                key = name.encode('ascii')
                info_keys.append(key)
                fields.remove(field)
                info_types[key] = types[field]
                info_numbers[key] = numbers[field]
                if field in fills:
                    info_fills[key] = fills[field]
        if info_keys:
            info_parser = VCFInfoParser(infos=info_keys,
                                        types=info_types,
                                        numbers=info_numbers,
                                        chunk_length=self.chunk_length,
                                        fills=info_fills)
        else:
            info_parser = VCFSkipFieldParser(key=b'INFO')
        info_parser.malloc_chunk()
        self.info_parser = info_parser

    def _init_format_calldata(self, fields, types, numbers, fills):
        """Setup FORMAT and calldata parsers."""
        format_keys = list()
        format_types = dict()
        format_numbers = dict()
        format_fills = dict()
        for field in list(fields):
            group, name = field.split('/')
            if group == 'calldata':
                key = name.encode('ascii')
                format_keys.append(key)
                fields.remove(field)
                format_types[key] = types[field]
                format_numbers[key] = numbers[field]
                if field in fills:
                    format_fills[key] = fills[field]
        if format_keys:
            format_parser = VCFFormatParser(formats=format_keys)
            calldata_parser = VCFCallDataParser(formats=format_keys,
                                                types=format_types,
                                                numbers=format_numbers,
                                                chunk_length=self.chunk_length,
                                                samples=self.samples,
                                                fills=format_fills)
        else:
            format_parser = VCFSkipFieldParser(key=b'FORMAT')
            calldata_parser = VCFSkipAllCallDataParser()
        format_parser.malloc_chunk()
        calldata_parser.malloc_chunk()
        self.format_parser = format_parser
        self.calldata_parser = calldata_parser

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:
        """Parse to end of current chunk or EOF."""

        with nogil:

            while True:

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
                    if context.chunk_variant_index + 1 == self.chunk_length:
                        # we're done
                        break

                elif context.state == VCFState.CHROM:
                    self.chrom_pos_parser.parse(stream, context)

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
        self.chrom_pos_parser.malloc_chunk()
        self.id_parser.malloc_chunk()
        self.ref_parser.malloc_chunk()
        self.alt_parser.malloc_chunk()
        self.qual_parser.malloc_chunk()
        self.filter_parser.malloc_chunk()
        self.info_parser.malloc_chunk()
        self.format_parser.malloc_chunk()
        self.calldata_parser.malloc_chunk()

    cdef object make_chunk(self, chunk_length):
        if chunk_length > 0:
            if chunk_length < self.chunk_length:
                limit = chunk_length
            else:
                limit = None
            chunk = dict()
            self.chrom_pos_parser.make_chunk(chunk, limit=limit)
            self.id_parser.make_chunk(chunk, limit=limit)
            self.ref_parser.make_chunk(chunk, limit=limit)
            self.alt_parser.make_chunk(chunk, limit=limit)
            self.qual_parser.make_chunk(chunk, limit=limit)
            self.filter_parser.make_chunk(chunk, limit=limit)
            self.info_parser.make_chunk(chunk, limit=limit)
            self.calldata_parser.make_chunk(chunk, limit=limit)
            return chunk

        else:
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:

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


cdef int vcf_read_field(InputStreamBase stream,
                        VCFContext context,
                        CharVector* dest) nogil except -1:

    # setup temp vector to store value
    CharVector_clear(dest)

    while True:

        if stream.c == 0:
            context.state = VCFState.EOF
            break

        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            break

        elif stream.c == TAB:
            stream.advance()
            # leave it to caller to advance state
            break

        else:
            CharVector_append(dest, stream.c)

        # advance input stream
        stream.advance()


cdef int vcf_parse_missing(InputStreamBase stream,
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
            stream.advance()
            context.state += 1
            break

        # advance input stream
        stream.advance()


cdef int vcf_skip_variant(InputStreamBase stream, VCFContext context) nogil except -1:
    # skip to EOL or EOF
    while True:
        if stream.c == 0:
            context.state = VCFState.EOF
            break
        elif stream.c == LF or stream.c == CR:
            context.state = VCFState.EOL
            break
        # advance input stream
        stream.advance()


cdef class VCFChromPosParser(VCFFieldParserBase):

    cdef:
        np.uint8_t[:] chrom_memory
        np.int32_t[:] pos_memory
        bint store_chrom
        bint store_pos
        char* region_chrom
        int region_begin
        int region_end
        np.ndarray chrom_values
        np.ndarray pos_values

    def __init__(self, dtype, store_chrom, store_pos, chunk_length, region_chrom,
                 region_begin, region_end):
        if store_chrom:
            dtype = check_string_dtype(dtype)
        super(VCFChromPosParser, self).__init__(key=b'CHROM', dtype=dtype, number=1,
                                                chunk_length=chunk_length)
        self.store_chrom = store_chrom
        self.store_pos = store_pos
        if region_chrom:
            self.region_chrom = PyBytes_AS_STRING(region_chrom)
            self.region_begin = region_begin
            self.region_end = region_end
        else:
            self.region_chrom = NULL
            self.region_begin = 0
            self.region_end = 0

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int i, n, cmp
            # index into memory view
            int memory_offset

        # reset context
        CharVector_clear(&context.chrom)
        context.pos = 0
        context.ref_len = 0

        # check for EOF early - important to handle file with no final line terminator
        if stream.c == 0:
            context.state = VCFState.EOF
            return 0

        # read chrom
        vcf_read_field(stream, context, &context.chrom)
        if context.chrom.size == 0:
            warn('empty CHROM', context)
        CharVector_terminate(&context.chrom)

        # read pos
        if context.state == VCFState.CHROM:
            context.state += 1
            # read pos
            vcf_read_field(stream, context, &context.temp)
            if context.temp.size == 0:
                warn('empty POS', context)
            else:
                vcf_strtol(&context.temp, context, &context.pos)

        if context.state == VCFState.POS:
            context.state += 1

        # handle region
        if self.region_chrom is not NULL:

            # compare with region chrom
            cmp = strcmp(context.chrom.data, self.region_chrom)

            if cmp < 0:
                return vcf_skip_variant(stream, context)

            if cmp > 0:
                # we're done
                context.state = EOF
                return 0

            if self.region_begin > 0 and context.pos < self.region_begin:
                return vcf_skip_variant(stream, context)

            if self.region_end > 0 and context.pos > self.region_end:
                # we're done
                context.state = EOF
                return 0

        # setup context
        context.sample_index = 0
        context.sample_field_index = 0
        context.sample_output_index = -1
        context.variant_index += 1
        context.chunk_variant_index += 1

        # store in chunk
        if self.store_chrom:

            # initialise memory index
            memory_offset = context.chunk_variant_index * self.itemsize

            # figure out how many characters to store
            n = min(context.chrom.size - 1, self.itemsize)

            # store characters
            for i in range(n):
                self.chrom_memory[memory_offset + i] = context.chrom.data[i]

        if self.store_pos:
            self.pos_memory[context.chunk_variant_index] = context.pos

    cdef int malloc_chunk(self) except -1:
        if self.store_chrom:
            self.chrom_values = np.zeros(self.chunk_length, dtype=self.dtype)
            self.chrom_memory = self.chrom_values.view('u1')
        if self.store_pos:
            self.pos_values = np.zeros(self.chunk_length, dtype='int32')
            self.pos_memory = self.pos_values

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store_chrom:
            chunk[CHROM_FIELD] = self.chrom_values[:limit]
        if self.store_pos:
            chunk[POS_FIELD] = self.pos_values[:limit]


cdef class VCFStringFieldParser(VCFFieldParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, key, dtype, chunk_length):
        dtype = check_string_dtype(dtype)
        super(VCFStringFieldParser, self).__init__(key=key, dtype=dtype, number=1,
                                                   chunk_length=chunk_length)

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


cdef class VCFRefParser(VCFFieldParserBase):

    cdef:
        np.uint8_t[:] memory
        bint store

    def __init__(self, dtype, chunk_length, store):
        if store:
            dtype = check_string_dtype(dtype)
        super(VCFRefParser, self).__init__(key=b'REF', dtype=dtype, number=1, chunk_length=chunk_length)
        self.store = store

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
                stream.advance()
                # advance to next field
                context.state += 1
                break

            else:
                context.ref_len += 1
                if self.store and chars_stored < self.itemsize:
                    # store value
                    self.memory[memory_index] = stream.c
                    # advance number of characters stored
                    chars_stored += 1
                    # advance memory index
                    memory_index += 1

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        if self.store:
            self.values = np.zeros(self.chunk_length, dtype=self.dtype)
            self.memory = self.values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store:
            chunk[REF_FIELD] = self.values[:limit]


cdef class VCFAltParser(VCFFieldParserBase):

    cdef:
        np.uint8_t[:] memory
        np.int32_t[:] numalt_memory
        np.int32_t[:, :] svlen_memory
        np.ndarray numalt_values
        np.ndarray svlen_values
        bint store_alt
        bint store_numalt
        bint store_svlen

    def __init__(self, dtype, number, chunk_length, store_alt, store_numalt, store_svlen):
        if store_alt:
            dtype = check_string_dtype(dtype)
        super(VCFAltParser, self).__init__(key=b'ALT', dtype=dtype, number=number, chunk_length=chunk_length)
        self.store_alt = store_alt
        self.store_numalt = store_numalt
        self.store_svlen = store_svlen

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            # index of alt values
            int alt_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0
            # size of alt allele relative to ref
            int svlen = -1 * context.ref_len

        # bail out early for missing value
        if stream.c == PERIOD:
            # treat period as missing value, regardless of what comes next
            return vcf_parse_missing(stream, context)

        # bail out early for empty value
        if stream.c == TAB:
            stream.advance()
            context.state += 1
            return 0

        # initialise memory offset and index
        memory_offset = context.chunk_variant_index * self.itemsize * self.number
        memory_index = memory_offset

        while True:

            if stream.c == 0:
                if self.store_svlen and alt_index < self.number:
                    self.svlen_memory[context.chunk_variant_index, alt_index] = svlen
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                if self.store_svlen and alt_index < self.number:
                    self.svlen_memory[context.chunk_variant_index, alt_index] = svlen
                context.state = VCFState.EOL
                break

            if stream.c == TAB:
                if self.store_svlen and alt_index < self.number:
                    self.svlen_memory[context.chunk_variant_index, alt_index] = svlen
                stream.advance()
                context.state += 1
                break

            elif stream.c == COMMA:
                if self.store_svlen and alt_index < self.number:
                    self.svlen_memory[context.chunk_variant_index, alt_index] = svlen
                # advance value index
                alt_index += 1
                # reset svlen
                svlen = -1 * context.ref_len
                # set memory index to beginning of next item
                memory_index = memory_offset + (alt_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            else:
                svlen += 1
                if self.store_alt and chars_stored < self.itemsize and alt_index < self.number:
                    # store value
                    self.memory[memory_index] = stream.c
                    # advance number of characters stored
                    chars_stored += 1
                    # advance memory index
                    memory_index += 1

            # advance input stream
            stream.advance()

        if self.store_numalt:
            self.numalt_memory[context.chunk_variant_index] = alt_index + 1

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        if self.store_alt:
            self.values = np.zeros(shape, dtype=self.dtype, order='C')
            self.memory = self.values.reshape(-1).view('u1')
        if self.store_numalt:
            self.numalt_values = np.zeros(self.chunk_length, dtype='int32')
            self.numalt_memory = self.numalt_values
        if self.store_svlen:
            self.svlen_values = np.zeros(shape, dtype='int32')
            self.svlen_memory = self.svlen_values

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store_alt:
            field = 'variants/' + str(self.key, 'ascii')
            values = self.values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]
        if self.store_numalt:
            field = 'variants/numalt'
            values = self.numalt_values
            chunk[field] = values[:limit]
        if self.store_svlen:
            field = 'variants/svlen'
            values = self.svlen_values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]


cdef class VCFQualParser(VCFFieldParserBase):

    cdef np.float32_t[:] memory

    def __init__(self, chunk_length, fill=NAN):
        super(VCFQualParser, self).__init__(key=b'QUAL', dtype='float32', number=1,
                                            fill=fill, chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            double value
            int parsed

        # read into temp
        vcf_read_field(stream, context, &context.temp)

        # parse string as floating
        parsed = vcf_strtod(&context.temp, context, &value)

        if parsed > 0:
            # store value
            self.memory[context.chunk_variant_index] = value

        if context.state == VCFState.QUAL:
            context.state += 1

    cdef int malloc_chunk(self) except -1:
        self.values = np.empty(self.chunk_length, dtype='float32')
        self.memory = self.values
        self.memory[:] = NAN


cdef class VCFFilterParser(VCFFieldParserBase):

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int filter_index

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return vcf_parse_missing(stream, context)

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

    cdef int parse_filter(self, VCFContext context) nogil except -1:
        cdef:
            int filter_index
            int i
            char* f

        if context.temp.size == 0:
            # not strictly kosher, treat as missing/empty
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
            chunk[field] = self.values[:limit, i]


##########################################################################################
# INFO parsing


cdef class VCFInfoParser(VCFFieldParserBase):

    cdef:
        tuple infos
        int n_infos
        char** infos_c
        tuple info_parsers
        PyObject** info_parsers_c
        VCFInfoParserBase skip_parser
        object fills

    def __cinit__(self, infos, types, numbers, chunk_length, fills):

        # setup INFO keys
        self.infos = tuple(sorted(infos))
        self.n_infos = len(self.infos)

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
                if t != np.dtype(bool):
                    warnings.warn('cannot have non-bool dtype for field with number 0, ignoring type %r' % t)
                parser = VCFInfoFlagParser(key, chunk_length=chunk_length)
            elif t == np.dtype('int8'):
                fill = fills.get(key, -1)
                parser = VCFInfoInt8Parser(key, number=n,
                                           fill=fill, chunk_length=chunk_length)
            elif t == np.dtype('int16'):
                fill = fills.get(key, -1)
                parser = VCFInfoInt16Parser(key, number=n, chunk_length=chunk_length,
                                            fill=fill)
            elif t == np.dtype('int32'):
                fill = fills.get(key, -1)
                parser = VCFInfoInt32Parser(key, number=n, chunk_length=chunk_length,
                                            fill=fill)
            elif t == np.dtype('int64'):
                fill = fills.get(key, -1)
                parser = VCFInfoInt64Parser(key, number=n, chunk_length=chunk_length,
                                            fill=fill)
            elif t == np.dtype('uint8'):
                fill = fills.get(key, IU8.max)
                parser = VCFInfoUInt8Parser(key, number=n, chunk_length=chunk_length,
                                            fill=fill)
            elif t == np.dtype('uint16'):
                fill = fills.get(key, IU16.max)
                parser = VCFInfoUInt16Parser(key, number=n, chunk_length=chunk_length,
                                             fill=fill)
            elif t == np.dtype('uint32'):
                fill = fills.get(key, IU32.max)
                parser = VCFInfoUInt32Parser(key, number=n, chunk_length=chunk_length,
                                             fill=fill)
            elif t == np.dtype('uint64'):
                fill = fills.get(key, IU64.max)
                parser = VCFInfoUInt64Parser(key, number=n, chunk_length=chunk_length,
                                             fill=fill)
            elif t == np.dtype('float32'):
                fill = fills.get(key, NAN)
                parser = VCFInfoFloat32Parser(key, number=n, chunk_length=chunk_length,
                                              fill=fill)
            elif t == np.dtype('float64'):
                fill = fills.get(key, NAN)
                parser = VCFInfoFloat64Parser(key, number=n, chunk_length=chunk_length,
                                              fill=fill)
            elif t == np.dtype(bool):
                parser = VCFInfoFlagParser(key, chunk_length=chunk_length)
            elif t.kind == 'S':
                parser = VCFInfoStringParser(key, dtype=t, number=n, chunk_length=chunk_length)
            else:
                parser = VCFInfoSkipParser(key)
                warnings.warn('type %s not supported for INFO field %r, field will be skipped' % (t, key))
            info_parsers.append(parser)
        self.info_parsers = tuple(info_parsers)

        # store pointers to parsers for nogil trickery
        self.info_parsers_c = <PyObject**> malloc(sizeof(PyObject*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_parsers_c[i] = <PyObject*> self.info_parsers[i]

    def __init__(self, infos, types, numbers, chunk_length, fills):
        super(VCFInfoParser, self).__init__(key=b'INFO', chunk_length=chunk_length)
        self.fills = fills

    def __dealloc__(self):
        if self.infos_c is not NULL:
            free(self.infos_c)
        if self.info_parsers_c is not NULL:
            free(self.info_parsers_c)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:

        # reset buffers
        CharVector_clear(&context.info_key)
        CharVector_clear(&context.info_val)

        # check for missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            return vcf_parse_missing(stream, context)

        # check for empty value
        if stream.c == TAB:
            # not strictly kosher, treat as missing/empty
            stream.advance()
            context.state += 1
            return 0

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

    cdef int parse_info(self,
                        InputStreamBase stream,
                        VCFContext context) nogil except -1:

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'variants/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values

    cdef int malloc_chunk(self) except -1:
        pass


cdef class VCFInfoInt8Parser(VCFInfoParserBase):

    cdef np.int8_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        kwargs.setdefault('fill', -1)
        super(VCFInfoInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoInt16Parser(VCFInfoParserBase):

    cdef np.int16_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        kwargs.setdefault('fill', -1)
        super(VCFInfoInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoInt32Parser(VCFInfoParserBase):

    cdef np.int32_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        kwargs.setdefault('fill', -1)
        super(VCFInfoInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoInt64Parser(VCFInfoParserBase):

    cdef np.int64_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        kwargs.setdefault('fill', -1)
        super(VCFInfoInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoUInt8Parser(VCFInfoParserBase):

    cdef np.uint8_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        kwargs.setdefault('fill', IU8.max)
        super(VCFInfoUInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoUInt16Parser(VCFInfoParserBase):

    cdef np.uint16_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint16'
        kwargs.setdefault('fill', IU16.max)
        super(VCFInfoUInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoUInt32Parser(VCFInfoParserBase):

    cdef np.uint32_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint32'
        kwargs.setdefault('fill', IU32.max)
        super(VCFInfoUInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoUInt64Parser(VCFInfoParserBase):

    cdef np.uint64_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint64'
        kwargs.setdefault('fill', IU64.max)
        super(VCFInfoUInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat32Parser(VCFInfoParserBase):

    cdef np.float32_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        kwargs.setdefault('fill', NAN)
        super(VCFInfoFloat32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFloat64Parser(VCFInfoParserBase):

    cdef np.float64_t[:, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        kwargs.setdefault('fill', NAN)
        super(VCFInfoFloat64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_info_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFInfoFlagParser(VCFInfoParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        super(VCFInfoFlagParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = check_string_dtype(kwargs.get('dtype'))
        super(VCFInfoStringParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

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

    def __init__(self, *args, **kwargs):
        super(VCFInfoSkipParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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
                                VCFContext context,
                                integer[:, :] memory) nogil except -1:
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


cdef int vcf_info_store_integer(VCFContext context,
                                int value_index,
                                integer[:, :] memory) nogil except -1:
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
                                 VCFContext context,
                                 floating[:, :] memory) nogil except -1:
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


cdef int vcf_info_store_floating(VCFContext context,
                                 int value_index,
                                 floating[:, :] memory) nogil except -1:
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

    def __init__(self, formats):
        super(VCFFormatParser, self).__init__(key=b'FORMAT')

    def __dealloc__(self):
        if self.formats_c is not NULL:
            free(self.formats_c)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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

    cdef int store_format(self, VCFContext context) nogil except -1:
        cdef int format_index

        # deal with empty or missing data
        if context.temp.size == 0:
            # not strictly kosher, handle as missing/empty
            return 0

        if context.temp.size == 1 and context.temp.data[0] == PERIOD:
            return 0

        # terminate the string
        CharVector_terminate(&context.temp)

        # find format index
        format_index = cstr_search_sorted(context.temp.data, self.formats_c, self.n_formats)

        # add to vector of indices for the current variant
        IntVector_append(&context.variant_format_indices, format_index)

        # clear out temp
        CharVector_clear(&context.temp)



cdef class VCFSkipAllCallDataParser(VCFFieldParserBase):
    """Skip a field."""

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_skip_variant(stream, context)

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


cdef class VCFCallDataParser(VCFFieldParserBase):

    cdef:
        tuple formats
        int n_formats
        tuple parsers
        PyObject** parsers_c
        VCFCallDataParserBase skip_parser
        np.uint8_t[:] samples
        int n_samples_out

    def __cinit__(self, formats, types, numbers, chunk_length, samples, fills):
        self.chunk_length = chunk_length
        self.samples = samples
        self.n_samples_out = np.count_nonzero(np.asarray(samples))

        # setup formats
        self.formats = tuple(sorted(formats))
        self.n_formats = len(self.formats)

        # setup parsers
        self.skip_parser = VCFCallDataSkipParser(key=None)
        parsers = list()
        kwds = dict(chunk_length=chunk_length, n_samples_out=self.n_samples_out)
        for key in self.formats:
            t = types[key]
            n = numbers[key]

            # special handling of "genotype" dtypes for any field
            if isinstance(t, str) and t.startswith('genotype/'):
                fill = fills.get(key, -1)
                t = np.dtype(t.split('/')[1])
                if t == np.dtype('int8'):
                    parser = VCFGenotypeInt8Parser(key, number=n, fill=fills.get(key, -1), **kwds)
                elif t == np.dtype('int16'):
                    parser = VCFGenotypeInt16Parser(key, number=n, fill=fills.get(key, -1), **kwds)
                elif t == np.dtype('int32'):
                    parser = VCFGenotypeInt32Parser(key, number=n, fill=fills.get(key, -1), **kwds)
                elif t == np.dtype('int64'):
                    parser = VCFGenotypeInt64Parser(key, number=n, fill=fills.get(key, -1), **kwds)
                elif t == np.dtype('uint8'):
                    parser = VCFGenotypeUInt8Parser(key, number=n, fill=fills.get(key, IU8.max), **kwds)
                elif t == np.dtype('uint16'):
                    parser = VCFGenotypeUInt16Parser(key, number=n, fill=fills.get(key, IU16.max), **kwds)
                elif t == np.dtype('uint32'):
                    parser = VCFGenotypeUInt32Parser(key, number=n, fill=fills.get(key, IU32.max), **kwds)
                elif t == np.dtype('uint64'):
                    parser = VCFGenotypeUInt64Parser(key, number=n, fill=fills.get(key, IU64.max), **kwds)
                else:
                    warnings.warn('type %r not supported for genotype field %r, field will be skipped' % (t, key))
                    parser = self.skip_parser

            # special handling of "genotype_ac" dtypes for any field
            elif isinstance(t, str) and t.startswith('genotype_ac/'):
                t = np.dtype(t.split('/')[1])
                if t == np.dtype('int8'):
                    parser = VCFGenotypeACInt8Parser(key, number=n, **kwds)
                elif t == np.dtype('int16'):
                    parser = VCFGenotypeACInt16Parser(key, number=n, **kwds)
                elif t == np.dtype('int32'):
                    parser = VCFGenotypeACInt32Parser(key, number=n, **kwds)
                elif t == np.dtype('int64'):
                    parser = VCFGenotypeACInt64Parser(key, number=n, **kwds)
                elif t == np.dtype('uint8'):
                    parser = VCFGenotypeACUInt8Parser(key, number=n, **kwds)
                elif t == np.dtype('uint16'):
                    parser = VCFGenotypeACUInt16Parser(key, number=n, **kwds)
                elif t == np.dtype('uint32'):
                    parser = VCFGenotypeACUInt32Parser(key, number=n, **kwds)
                elif t == np.dtype('uint64'):
                    parser = VCFGenotypeACUInt64Parser(key, number=n, **kwds)
                else:
                    warnings.warn('type %r not supported for genotype_ac field %r, field will be skipped' % (t, key))
                    parser = self.skip_parser

            # special handling of GT field
            elif key == b'GT' and t == np.dtype('int8'):
                parser = VCFGenotypeInt8Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif key == b'GT' and t == np.dtype('int16'):
                parser = VCFGenotypeInt16Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif key == b'GT' and t == np.dtype('int32'):
                parser = VCFGenotypeInt32Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif key == b'GT' and t == np.dtype('int64'):
                parser = VCFGenotypeInt64Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif key == b'GT' and t == np.dtype('uint8'):
                parser = VCFGenotypeUInt8Parser(key, number=n, fill=fills.get(key, IU8.max), **kwds)
            elif key == b'GT' and t == np.dtype('uint16'):
                parser = VCFGenotypeUInt16Parser(key, number=n, fill=fills.get(key, IU16.max), **kwds)
            elif key == b'GT' and t == np.dtype('uint32'):
                parser = VCFGenotypeUInt32Parser(key, number=n, fill=fills.get(key, IU32.max), **kwds)
            elif key == b'GT' and t == np.dtype('uint64'):
                parser = VCFGenotypeUInt64Parser(key, number=n, fill=fills.get(key, IU64.max), **kwds)

            # all other calldata
            elif t == np.dtype('int8'):
                parser = VCFCallDataInt8Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif t == np.dtype('int16'):
                parser = VCFCallDataInt16Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif t == np.dtype('int32'):
                parser = VCFCallDataInt32Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif t == np.dtype('int64'):
                parser = VCFCallDataInt64Parser(key, number=n, fill=fills.get(key, -1), **kwds)
            elif t == np.dtype('uint8'):
                parser = VCFCallDataUInt8Parser(key, number=n, fill=fills.get(key, IU8.max), **kwds)
            elif t == np.dtype('uint16'):
                parser = VCFCallDataUInt16Parser(key, number=n, fill=fills.get(key, IU16.max), **kwds)
            elif t == np.dtype('uint32'):
                parser = VCFCallDataUInt32Parser(key, number=n, fill=fills.get(key, IU32.max), **kwds)
            elif t == np.dtype('uint64'):
                parser = VCFCallDataUInt64Parser(key, number=n, fill=fills.get(key, IU64.max), **kwds)
            elif t == np.dtype('float32'):
                parser = VCFCallDataFloat32Parser(key, number=n, fill=fills.get(key, NAN), **kwds)
            elif t == np.dtype('float64'):
                parser = VCFCallDataFloat64Parser(key, number=n, fill=fills.get(key, NAN), **kwds)
            elif t.kind == 'S':
                parser = VCFCallDataStringParser(key, dtype=t, number=n, **kwds)

            else:
                parser = VCFCallDataSkipParser(key)
                warnings.warn('type %r not supported for FORMAT field %r, field will be skipped' % (t, key))

            parsers.append(parser)
        self.parsers = tuple(parsers)

        # store pointers to parsers
        self.parsers_c = <PyObject**> malloc(sizeof(PyObject*) * self.n_formats)
        for i in range(self.n_formats):
            self.parsers_c[i] = <PyObject*> self.parsers[i]

    def __init__(self, formats, types, numbers, chunk_length, samples, fills):
        super(VCFCallDataParser, self).__init__(chunk_length=chunk_length)

    def __dealloc__(self):
        if self.parsers_c is not NULL:
            free(self.parsers_c)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        cdef:
            int i
            PyObject* parser

        # initialise context
        context.sample_index = 0
        context.sample_output_index = -1
        context.sample_field_index = 0

        # setup output indexing
        if self.samples[0]:
            context.sample_output_index += 1
        else:
            # skip to next sample
            while stream.c != 0 and stream.c != LF and stream.c != CR and stream.c != TAB:
                stream.advance()

        while True:

            if stream.c == 0:
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                context.state = VCFState.EOL
                break

            elif stream.c == TAB:
                stream.advance()
                context.sample_index += 1
                context.sample_field_index = 0
                if self.samples[context.sample_index]:
                    context.sample_output_index += 1
                else:
                    # skip to next sample
                    while stream.c != 0 and stream.c != LF and stream.c != CR and stream.c != TAB:
                        stream.advance()

            elif context.sample_index >= self.samples.shape[0]:
                # more samples than we expected, skip to EOL
                while stream.c != 0 and stream.c != LF and stream.c != CR:
                    stream.advance()

            elif stream.c == COLON:
                context.sample_field_index += 1
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
        int n_samples_out

    def __init__(self, key=None, dtype=None, number=1, fill=0, chunk_length=0,
                 n_samples_out=0):
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
        self.n_samples_out = n_samples_out

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'calldata/' + str(self.key, 'ascii')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        pass


cdef class VCFCallDataSkipParser(VCFCallDataParserBase):

    def __init__(self, key, *args, **kwargs):
        super(VCFCallDataSkipParser, self).__init__(key=key)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
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
        kwargs.setdefault('fill', -1)
        super(VCFGenotypeInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt16Parser(VCFCallDataParserBase):

    cdef:
        np.int16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        kwargs.setdefault('fill', -1)
        super(VCFGenotypeInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt32Parser(VCFCallDataParserBase):

    cdef:
        np.int32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        kwargs.setdefault('fill', -1)
        super(VCFGenotypeInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeInt64Parser(VCFCallDataParserBase):

    cdef:
        np.int64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        kwargs.setdefault('fill', -1)
        super(VCFGenotypeInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeUInt8Parser(VCFCallDataParserBase):

    cdef:
        np.uint8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        kwargs.setdefault('fill', IU8.max)
        super(VCFGenotypeUInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeUInt16Parser(VCFCallDataParserBase):

    cdef:
        np.uint16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint16'
        kwargs.setdefault('fill', IU16.max)
        super(VCFGenotypeUInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeUInt32Parser(VCFCallDataParserBase):

    cdef:
        np.uint32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint32'
        kwargs.setdefault('fill', IU32.max)
        super(VCFGenotypeUInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFGenotypeUInt64Parser(VCFCallDataParserBase):

    cdef:
        np.uint64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint64'
        kwargs.setdefault('fill', IU64.max)
        super(VCFGenotypeUInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_genotype_parse(InputStreamBase stream,
                            VCFContext context,
                            integer[:, :, :] memory) nogil except -1:
    cdef:
        int value_index = 0

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


cdef int vcf_genotype_store(VCFContext context,
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
        memory[context.chunk_variant_index, context.sample_output_index, value_index] = allele


cdef class VCFGenotypeACInt8Parser(VCFCallDataParserBase):

    cdef:
        np.int8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        kwargs['fill'] = 0
        super(VCFGenotypeACInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACInt16Parser(VCFCallDataParserBase):

    cdef:
        np.int16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        kwargs['fill'] = 0
        super(VCFGenotypeACInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACInt32Parser(VCFCallDataParserBase):

    cdef:
        np.int32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        kwargs['fill'] = 0
        super(VCFGenotypeACInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACInt64Parser(VCFCallDataParserBase):

    cdef:
        np.int64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        kwargs['fill'] = 0
        super(VCFGenotypeACInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACUInt8Parser(VCFCallDataParserBase):

    cdef:
        np.uint8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        kwargs['fill'] = 0
        super(VCFGenotypeACUInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACUInt16Parser(VCFCallDataParserBase):

    cdef:
        np.uint16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint16'
        kwargs['fill'] = 0
        super(VCFGenotypeACUInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACUInt32Parser(VCFCallDataParserBase):

    cdef:
        np.uint32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint32'
        kwargs['fill'] = 0
        super(VCFGenotypeACUInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef class VCFGenotypeACUInt64Parser(VCFCallDataParserBase):

    cdef:
        np.uint64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint64'
        kwargs['fill'] = 0
        super(VCFGenotypeACUInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef int vcf_genotype_ac_parse(InputStreamBase stream,
                               VCFContext context,
                               integer[:, :, :] memory) nogil except -1:
    # reset temporary buffer
    CharVector_clear(&context.temp)

    while True:

        if stream.c == SLASH or stream.c == PIPE:
            vcf_genotype_ac_store(context, memory)
            CharVector_clear(&context.temp)

        elif stream.c == COLON or \
                stream.c == TAB or \
                stream.c == LF or \
                stream.c == CR or \
                stream.c == 0:
            vcf_genotype_ac_store(context, memory)
            break

        else:
            CharVector_append(&context.temp, stream.c)

        stream.advance()


cdef int vcf_genotype_ac_store(VCFContext context,
                               integer[:, :, :] memory) nogil except -1:
    cdef:
        int parsed
        long allele

    # attempt to parse allele
    parsed = vcf_strtol(&context.temp, context, &allele)

    # store value
    if parsed > 0 and allele < memory.shape[2]:
        memory[context.chunk_variant_index, context.sample_output_index, allele] += 1


cdef class VCFCallDataInt8Parser(VCFCallDataParserBase):

    cdef np.int8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        kwargs.setdefault('fill', -1)
        super(VCFCallDataInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt16Parser(VCFCallDataParserBase):

    cdef np.int16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int16'
        kwargs.setdefault('fill', -1)
        super(VCFCallDataInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt32Parser(VCFCallDataParserBase):

    cdef np.int32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        kwargs.setdefault('fill', -1)
        super(VCFCallDataInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataInt64Parser(VCFCallDataParserBase):

    cdef np.int64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        kwargs.setdefault('fill', -1)
        super(VCFCallDataInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataUInt8Parser(VCFCallDataParserBase):

    cdef np.uint8_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        kwargs.setdefault('fill', IU8.max)
        super(VCFCallDataUInt8Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataUInt16Parser(VCFCallDataParserBase):

    cdef np.uint16_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint16'
        kwargs.setdefault('fill', IU16.max)
        super(VCFCallDataUInt16Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataUInt32Parser(VCFCallDataParserBase):

    cdef np.uint32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint32'
        kwargs.setdefault('fill', IU32.max)
        super(VCFCallDataUInt32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataUInt64Parser(VCFCallDataParserBase):

    cdef np.uint64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'uint64'
        kwargs.setdefault('fill', IU64.max)
        super(VCFCallDataUInt64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_integer(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataFloat32Parser(VCFCallDataParserBase):

    cdef np.float32_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        kwargs.setdefault('fill', NAN)
        super(VCFCallDataFloat32Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef class VCFCallDataFloat64Parser(VCFCallDataParserBase):

    cdef np.float64_t[:, :, :] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        kwargs.setdefault('fill', NAN)
        super(VCFCallDataFloat64Parser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) nogil except -1:
        return vcf_calldata_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_calldata_parse_integer(InputStreamBase stream,
                                    VCFContext context,
                                    integer[:, :, :] memory) nogil except -1:

    cdef:
        int value_index = 0

    # reset temporary buffer
    CharVector_clear(&context.temp)

    while True:

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


cdef int vcf_calldata_store_integer(VCFContext context,
                                    int value_index,
                                    integer[:, :, :] memory) nogil except -1:
    cdef:
        int parsed
        long value

    if value_index >= memory.shape[2]:
        # more values than we have room for, ignore
        return 0

    parsed = vcf_strtol(&context.temp, context, &value)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_output_index, value_index] = value


cdef int vcf_calldata_parse_floating(InputStreamBase stream,
                                     VCFContext context,
                                     floating[:, :, :] memory) nogil except -1:

    cdef:
        int value_index = 0

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


cdef int vcf_calldata_store_floating(VCFContext context,
                                     int value_index,
                                     floating[:, :, :] memory) nogil except -1:
    cdef:
        int parsed
        double value

    if value_index >= memory.shape[2]:
        # more values than we have room for, ignore
        return 0

    parsed = vcf_strtod(&context.temp, context, &value)

    # store value
    if parsed > 0:
        memory[context.chunk_variant_index, context.sample_output_index, value_index] = value


cdef class VCFCallDataStringParser(VCFCallDataParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = check_string_dtype(kwargs.get('dtype'))
        super(VCFCallDataStringParser, self).__init__(*args, **kwargs)

    cdef int parse(self,
                   InputStreamBase stream,
                   VCFContext context) nogil except -1:
        cdef:
            int value_index = 0
            # index into memory view
            int memory_offset, memory_index
            # number of characters read into current value
            int chars_stored = 0

        # initialise memory index
        memory_offset = ((context.chunk_variant_index *
                         self.n_samples_out *
                         self.number *
                         self.itemsize) +
                         (context.sample_output_index *
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
        shape = (self.chunk_length, self.n_samples_out, self.number)
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


cdef int vcf_strtol(CharVector* value, VCFContext context, long* l) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if value.size == 0:
        # not strictly kosher, treat as missing value
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


cdef int vcf_strtod(CharVector* value, VCFContext context, double* d) nogil except -1:
    cdef:
        char* str_end
        int parsed

    if value.size == 0:
        # not strictly kosher, treat as missing value
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


vcf_state_labels = [
    'CHROM',
    'POS',
    'ID',
    'REF',
    'ALT',
    'QUAL',
    'FILTER',
    'INFO',
    'FORMAT',
    'CALLDATA',
    'EOL',
    'EOF'
]


cdef int warn(message, VCFContext context) nogil except -1:
    with gil:
        message += '; field: %s' % vcf_state_labels[context.state]
        message += '; variant: %s' % context.variant_index
        if context.state > VCFState.POS:
            message += ' (%s:%s)' % (str(CharVector_to_pybytes(&context.chrom), 'ascii'), context.pos)
        if context.state == VCFState.CALLDATA:
            message += '; sample: %s:%s (%s)' % (context.sample_index, context.sample_field_index,
                                                 context.headers.samples[context.sample_index])
        warnings.warn(message)


# cdef int debug(message, vars=None) except -1:
#     message = '[DEBUG] ' + str(message)
#     if vars:
#         message = message % vars
#     print(message, file=sys.stderr)
#     sys.stderr.flush()


##########################################################################################
# EXPERIMENTAL support for multi-threaded parsing


import itertools
import time


cdef class VCFParallelParser:

    cdef:
        FileInputStream stream
        CharVectorInputStream buffer
        VCFContext context
        VCFParser parser
        int chunk_length
        int block_length
        object pool
        object result

    def __cinit__(self, stream, parser, chunk_length, block_length, pool, headers):
        self.buffer = CharVectorInputStream(2**14)
        self.context = VCFContext(headers)
        self.stream = stream
        self.parser = parser
        self.chunk_length = chunk_length
        self.block_length = block_length
        self.pool = pool
        self.result = None

    def read(self, n_lines):
        self.buffer.clear()
        n_lines_read = self.stream.read_lines_into(&(self.buffer.vector), n_lines)
        self.buffer.advance()
        return n_lines_read

    def parse_async(self, block_index, chunk_index):
        self.result = self.pool.apply_async(self.parse, args=(block_index, chunk_index))

    def join(self):
        if self.result is not None:
            self.result.get()

    def parse(self, block_index, chunk_index):
        before = time.time()
        # set initial state
        self.context.state = VCFState.CHROM
        self.context.chunk_variant_index = block_index * self.block_length - 1
        self.context.variant_index = (chunk_index * self.chunk_length +
                                      self.context.chunk_variant_index)
        # parse the block of data stored in the buffer
        self.parser.parse(self.buffer, self.context)
        after = time.time()


cdef class VCFParallelChunkIterator:

    cdef:
        FileInputStream stream
        VCFParser parser
        object pool
        int chunk_length
        int block_length
        int n_threads
        int n_workers
        int chunk_index
        list workers

    def __cinit__(self,
                  FileInputStream stream,
                  int chunk_length, int block_length, int n_threads,
                  headers, fields, types, numbers, fills, region, samples):

        self.stream = stream
        self.chunk_length = chunk_length
        self.n_threads = n_threads
        self.pool = ThreadPool(n_threads)
        # allow one more worker than number of threads in pool to allow for sync
        # reading of data in the main thread
        self.n_workers = n_threads + 1
        # only makes sense to have block length at most fraction chunk length if we want
        # some parallelism
        self.block_length = min(block_length, chunk_length//self.n_workers)
        if self.block_length < 1:
            self.block_length = 1
        samples = check_samples(samples, headers)
        self.parser = VCFParser(
            fields=fields, types=types, numbers=numbers, chunk_length=chunk_length, samples=samples, fills=fills,
            region=region
        )
        self.chunk_index = -1
        self.workers = [VCFParallelParser(stream=stream, parser=self.parser, chunk_length=self.chunk_length,
                                          block_length=self.block_length, pool=self.pool, headers=headers)
                        for _ in range(self.n_workers)]

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            int block_index = 0
            int i = 0
            int n_lines
            int n_lines_read = 0
            VCFParallelParser worker

        # increment the current chunk index
        self.chunk_index += 1

        # allocate arrays for next chunk
        self.parser.malloc_chunk()

        # cycle around the workers
        for i in itertools.cycle(list(range(self.n_workers))):
            worker = self.workers[i]

            # wait for the result to finish - this ensures we don't overwrite a
            # worker's buffer while it's still parsing
            worker.join()

            # read lines into the worker's buffer - this part has to be synchronous
            n_lines = min(self.block_length, self.chunk_length - n_lines_read)
            n_lines_read += worker.read(n_lines)

            # launch parsing of the block in parallel
            worker.parse_async(block_index, self.chunk_index)

            # increment the current block index
            block_index += 1

            # is the chunk done?
            if n_lines_read >= self.chunk_length:
                break

            # is the input stream exhausted?
            if self.stream.c == 0:
                break

        # wait for all parallel tasks to complete
        for worker in self.workers:
            worker.join()

        # obtain the final chunk length via the last worker
        worker = self.workers[i]
        chunk_length = worker.context.chunk_variant_index + 1

        # obtain the chunk
        chunk = self.parser.make_chunk(chunk_length)

        if chunk is None:
            # clean up thread pool
            self.pool.close()
            self.pool.join()
            self.pool.terminate()
            raise StopIteration

        else:
            chrom = CharVector_to_pybytes(&worker.context.chrom)
            pos = worker.context.pos
            return chunk, chunk_length, chrom, pos


###################################################################
# ANN transformer


cdef enum ANNFidx:
    ALLELE
    ANNOTATION
    ANNOTATION_IMPACT
    GENE_NAME
    GENE_ID
    FEATURE_TYPE
    FEATURE_ID
    TRANSCRIPT_BIOTYPE
    RANK
    HGVS_C
    HGVS_P
    CDNA
    CDS
    AA
    DISTANCE


ANN_FIELD = 'variants/ANN'
ANN_ALLELE_FIELD = 'variants/ANN_Allele'
ANN_ANNOTATION_FIELD = 'variants/ANN_Annotation'
ANN_ANNOTATION_IMPACT_FIELD = 'variants/ANN_Annotation_Impact'
ANN_GENE_NAME_FIELD = 'variants/ANN_Gene_Name'
ANN_GENE_ID_FIELD = 'variants/ANN_Gene_ID'
ANN_FEATURE_TYPE_FIELD = 'variants/ANN_Feature_Type'
ANN_FEATURE_ID_FIELD = 'variants/ANN_Feature_ID'
ANN_TRANSCRIPT_BIOTYPE_FIELD = 'variants/ANN_Transcript_BioType'
ANN_RANK_FIELD = 'variants/ANN_Rank'
ANN_HGVS_C_FIELD = 'variants/ANN_HGVS_c'
ANN_HGVS_P_FIELD = 'variants/ANN_HGVS_p'
ANN_CDNA_FIELD = 'variants/ANN_cDNA'
ANN_CDS_FIELD = 'variants/ANN_CDS'
ANN_AA_FIELD = 'variants/ANN_AA'
ANN_DISTANCE_FIELD = 'variants/ANN_Distance'


ANN_FIELDS = (
    ANN_ALLELE_FIELD,
    ANN_ANNOTATION_FIELD,
    ANN_ANNOTATION_IMPACT_FIELD,
    ANN_GENE_NAME_FIELD,
    ANN_GENE_ID_FIELD,
    ANN_FEATURE_TYPE_FIELD,
    ANN_FEATURE_ID_FIELD,
    ANN_TRANSCRIPT_BIOTYPE_FIELD,
    ANN_RANK_FIELD,
    ANN_HGVS_C_FIELD,
    ANN_HGVS_P_FIELD,
    ANN_CDNA_FIELD,
    ANN_CDS_FIELD,
    ANN_AA_FIELD,
    ANN_DISTANCE_FIELD
)


def _normalize_ann_field_prefix(f):
    # normalize prefix
    if f.startswith('variants/ANN_'):
        pass
    elif f.startswith('ANN_'):
        f = 'variants/' + f
    else:
        f = 'variants/ANN_' + f
    if f not in ANN_FIELDS:
        warnings.warn('invalid ANN field %r, will be ignored' % f)
        f = None
    return f


def _normalize_ann_fields(fields):
    normed_fields = set()

    if fields is None:
        return set(ANN_FIELDS)

    else:
        for f in fields:
            f = _normalize_ann_field_prefix(f)
            if f:
                normed_fields.add(f)

    return normed_fields


default_ann_types = dict()
default_ann_types[ANN_ALLELE_FIELD] = np.dtype('S1')
default_ann_types[ANN_ANNOTATION_FIELD] = np.dtype('S34')
default_ann_types[ANN_ANNOTATION_IMPACT_FIELD] = np.dtype('S8')
default_ann_types[ANN_GENE_NAME_FIELD] = np.dtype('S14')
default_ann_types[ANN_GENE_ID_FIELD] = np.dtype('S14')
default_ann_types[ANN_FEATURE_TYPE_FIELD] = np.dtype('S20')
default_ann_types[ANN_FEATURE_ID_FIELD] = np.dtype('S14')
default_ann_types[ANN_TRANSCRIPT_BIOTYPE_FIELD] = np.dtype('S20')
default_ann_types[ANN_RANK_FIELD] = np.dtype('int8')
default_ann_types[ANN_HGVS_C_FIELD] = np.dtype('S16')
default_ann_types[ANN_HGVS_P_FIELD] = np.dtype('S16')
default_ann_types[ANN_CDNA_FIELD] = np.dtype('int32')
default_ann_types[ANN_CDS_FIELD] = np.dtype('int32')
default_ann_types[ANN_AA_FIELD] = np.dtype('int32')
default_ann_types[ANN_DISTANCE_FIELD] = np.dtype('int32')


def _normalize_ann_types(fields, types):

    if types is None:
        types = dict()
    types = {_normalize_ann_field_prefix(f): np.dtype(t)
             for f, t in types.items()}

    normed_types = dict()
    for f in fields:
        if f in types:
            normed_types[f] = types[f]
        else:
            normed_types[f] = default_ann_types[f]

    return normed_types


cdef class ANNTransformer:

    cdef:
        set fields
        object types
        bint keep_original
        np.uint8_t[:] emit

    def __init__(self, fields=None, types=None, keep_original=False):
        self.fields = _normalize_ann_fields(fields)
        self.types = _normalize_ann_types(self.fields, types)
        self.keep_original = keep_original

    def transform(self, chunk):
        cdef:
            int i, j, chunk_length, number
            bytes raw
            list vals
            bytes v
            list vv

        # obtain array to be transformed
        ann = chunk[ANN_FIELD]
        if not self.keep_original:
            del chunk[ANN_FIELD]

        # determine chunk length and number of items
        chunk_length = ann.shape[0]
        if ann.ndim == 1:
            ann = ann[:, np.newaxis]
        number = ann.shape[1]
        shape = chunk_length, number

        # allocate output arrays
        if ANN_ALLELE_FIELD in self.fields:
            allele = np.zeros(shape, dtype=self.types[ANN_ALLELE_FIELD])
        else:
            allele = None
        if ANN_ANNOTATION_FIELD in self.fields:
            annotation = np.zeros(shape, dtype=self.types[ANN_ANNOTATION_FIELD])
        else:
            annotation = None
        if ANN_ANNOTATION_IMPACT_FIELD in self.fields:
            annotation_impact = np.zeros(shape, dtype=self.types[ANN_ANNOTATION_IMPACT_FIELD])
        else:
            annotation_impact = None
        if ANN_GENE_NAME_FIELD in self.fields:
            gene_name = np.zeros(shape, dtype=self.types[ANN_GENE_NAME_FIELD])
        else:
            gene_name = None
        if ANN_GENE_ID_FIELD in self.fields:
            gene_id = np.zeros(shape, dtype=self.types[ANN_GENE_ID_FIELD])
        else:
            gene_id = None
        if ANN_FEATURE_TYPE_FIELD in self.fields:
            feature_type = np.zeros(shape, dtype=self.types[ANN_FEATURE_TYPE_FIELD])
        else:
            feature_type = None
        if ANN_FEATURE_ID_FIELD in self.fields:
            feature_id = np.zeros(shape, dtype=self.types[ANN_FEATURE_ID_FIELD])
        else:
            feature_id = None
        if ANN_TRANSCRIPT_BIOTYPE_FIELD in self.fields:
            transcript_biotype = np.zeros(shape, dtype=self.types[ANN_TRANSCRIPT_BIOTYPE_FIELD])
        else:
            transcript_biotype = None
        if ANN_RANK_FIELD in self.fields:
            rank = np.empty(shape + (2,), dtype=self.types[ANN_RANK_FIELD])
            rank.fill(-1)
        else:
            rank = None
        if ANN_HGVS_C_FIELD in self.fields:
            hgvs_c = np.zeros(shape, dtype=self.types[ANN_HGVS_C_FIELD])
        else:
            hgvs_c = None
        if ANN_HGVS_P_FIELD in self.fields:
            hgvs_p = np.zeros(shape, dtype=self.types[ANN_HGVS_P_FIELD])
        else:
            hgvs_p = None
        if ANN_CDNA_FIELD in self.fields:
            cdna = np.empty(shape + (2,), dtype=self.types[ANN_CDNA_FIELD])
            cdna.fill(-1)
        else:
            cdna = None
        if ANN_CDS_FIELD in self.fields:
            cds = np.empty(shape + (2,), dtype=self.types[ANN_CDS_FIELD])
            cds.fill(-1)
        else:
            cds = None
        if ANN_AA_FIELD in self.fields:
            aa = np.empty(shape + (2,), dtype=self.types[ANN_AA_FIELD])
            aa.fill(-1)
        else:
            aa = None
        if ANN_DISTANCE_FIELD in self.fields:
            distance = np.empty(shape, dtype=self.types[ANN_DISTANCE_FIELD])
            distance.fill(-1)
        else:
            distance = None

        # start working
        for i in range(chunk_length):
            for j in range(number):

                # obtain raw value
                raw = <bytes>ann[i, j]

                # bail early if no content
                if raw == b'' or raw == b'.':
                    continue

                # split fields
                vals = raw.split(b'|')

                # convert and store values
                try:
                    if allele is not None:
                        v = vals[ANNFidx.ALLELE]
                        if v:
                            allele[i, j] = v
                    if annotation is not None:
                        v = vals[ANNFidx.ANNOTATION]
                        if v:
                            annotation[i, j] = v
                    if annotation_impact is not None:
                        v = vals[ANNFidx.ANNOTATION_IMPACT]
                        if v:
                            annotation_impact[i, j] = v
                    if gene_name is not None:
                        v = vals[ANNFidx.GENE_NAME]
                        if v:
                            gene_name[i, j] = v
                    if gene_id is not None:
                        v = vals[ANNFidx.GENE_ID]
                        if v:
                            gene_id[i, j] = v
                    if feature_type is not None:
                        v = vals[ANNFidx.FEATURE_TYPE]
                        if v:
                            feature_type[i, j] = v
                    if feature_id is not None:
                        v = vals[ANNFidx.FEATURE_ID]
                        if v:
                            feature_id[i, j] = v
                    if transcript_biotype is not None:
                        v = vals[ANNFidx.TRANSCRIPT_BIOTYPE]
                        if v:
                            transcript_biotype[i, j] = v
                    if rank is not None:
                        v = vals[ANNFidx.RANK]
                        if v:
                            vv = v.split(b'/')
                            rank[i, j, 0] = int(vv[0])
                            rank[i, j, 1] = int(vv[1])
                    if hgvs_c is not None:
                        v = vals[ANNFidx.HGVS_C]
                        if v:
                            hgvs_c[i, j] = v[2:]
                    if hgvs_p is not None:
                        v = vals[ANNFidx.HGVS_P]
                        if v:
                            hgvs_p[i, j] = v[2:]
                    if cdna is not None:
                        v = vals[ANNFidx.CDNA]
                        if v:
                            vv = v.split(b'/')
                            cdna[i, j, 0] = int(vv[0])
                            cdna[i, j, 1] = int(vv[1])
                    if cds is not None:
                        v = vals[ANNFidx.CDS]
                        if v:
                            vv = v.split(b'/')
                            cds[i, j, 0] = int(vv[0])
                            cds[i, j, 1] = int(vv[1])
                    if aa is not None:
                        v = vals[ANNFidx.AA]
                        if v:
                            vv = v.split(b'/')
                            aa[i, j, 0] = int(vv[0])
                            aa[i, j, 1] = int(vv[1])
                    if distance is not None:
                        v = vals[ANNFidx.DISTANCE]
                        if v:
                            distance[i, j] = v

                except IndexError:
                    warnings.warn('missing fields in ANN value')

        ann_chunk = dict()
        if allele is not None:
            ann_chunk[ANN_ALLELE_FIELD] = allele
        if annotation is not None:
            ann_chunk[ANN_ANNOTATION_FIELD] = annotation
        if annotation_impact is not None:
            ann_chunk[ANN_ANNOTATION_IMPACT_FIELD] = annotation_impact
        if gene_name is not None:
            ann_chunk[ANN_GENE_NAME_FIELD] = gene_name
        if gene_id is not None:
            ann_chunk[ANN_GENE_ID_FIELD] = gene_id
        if feature_type is not None:
            ann_chunk[ANN_FEATURE_TYPE_FIELD] = feature_type
        if feature_id is not None:
            ann_chunk[ANN_FEATURE_ID_FIELD] = feature_id
        if transcript_biotype is not None:
            ann_chunk[ANN_TRANSCRIPT_BIOTYPE_FIELD] = transcript_biotype
        if rank is not None:
            ann_chunk[ANN_RANK_FIELD] = rank
        if hgvs_c is not None:
            ann_chunk[ANN_HGVS_C_FIELD] = hgvs_c
        if hgvs_p is not None:
            ann_chunk[ANN_HGVS_P_FIELD] = hgvs_p
        if cdna is not None:
            ann_chunk[ANN_CDNA_FIELD] = cdna
        if cds is not None:
            ann_chunk[ANN_CDS_FIELD] = cds
        if aa is not None:
            ann_chunk[ANN_AA_FIELD] = aa
        if distance is not None:
            ann_chunk[ANN_DISTANCE_FIELD] = distance

        if number == 1:
            for k in list(ann_chunk.keys()):
                ann_chunk[k] = ann_chunk[k].squeeze(axis=1)

        chunk.update(ann_chunk)
