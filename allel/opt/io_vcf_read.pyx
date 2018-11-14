# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=2
"""
# options for profiling...
# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
from __future__ import absolute_import, print_function, division


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
# from multiprocessing.pool import ThreadPool


from allel.compat import PY2, text_type


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
cdef char ASTERISK = b'*'

# user field specifications for fixed fields
CHROM_FIELD = 'variants/CHROM'
POS_FIELD = 'variants/POS'
ID_FIELD = 'variants/ID'
REF_FIELD = 'variants/REF'
ALT_FIELD = 'variants/ALT'
QUAL_FIELD = 'variants/QUAL'
NUMALT_FIELD = 'variants/numalt'
ALTLEN_FIELD = 'variants/altlen'
IS_SNP_FIELD = 'variants/is_snp'

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
    Py_ssize_t size
    Py_ssize_t capacity
    char* data


cdef inline void CharVector_init(CharVector* self, Py_ssize_t capacity) nogil:
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


cdef object CharVector_to_pystr(CharVector* self):
    v = PyBytes_FromStringAndSize(self.data, self.size)
    v = text_type(v, 'utf8')
    return v


cdef object CharVector_to_pystr_sized(CharVector* self, Py_ssize_t size):
    v = PyBytes_FromStringAndSize(self.data, size)
    v = text_type(v, 'utf8')
    return v


cdef struct IntVector:
    Py_ssize_t size
    Py_ssize_t capacity
    int* data


cdef inline void IntVector_init(IntVector* self, Py_ssize_t capacity) nogil:
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


##########################################################################################
# C string utilities.


cdef inline Py_ssize_t search_sorted_cstr(char* query, char** compare, Py_ssize_t n_items) nogil:
    cdef:
        Py_ssize_t i

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

    cdef int advance(self) except -1:  # nogil
        """Read the next character from the stream and store it in the `c` attribute."""
        pass


cdef class FileInputStream(InputStreamBase):

    cdef:
        # Python file-like object
        object fileobj
        Py_ssize_t buffer_size
        bytearray buffer
        char* buffer_start
        char* buffer_end
        char* stream
        bint close

    def __init__(self, fileobj, buffer_size=2**14, close=False):
        self.fileobj = fileobj
        self.buffer_size = buffer_size
        # initialise input buffer
        self.buffer = bytearray(buffer_size)
        self.buffer_start = PyByteArray_AS_STRING(self.buffer)
        self.stream = self.buffer_start
        self.close = close
        self._bufferup()
        self.advance()

    def __dealloc__(self):
        if self.close:
            self.fileobj.close()

    cdef int _bufferup(self) except -1:  # nogil
        """Read as many bytes as possible from the underlying file object into the
        buffer."""
        cdef Py_ssize_t l
        # with gil:
        l = self.fileobj.readinto(self.buffer)
        if l > 0:
            self.stream = self.buffer_start
            self.buffer_end = self.buffer_start + l
        else:
            self.stream = NULL

    cdef int advance(self) except -1:  # nogil
        """Read the next character from the stream and store it in the `c` attribute."""
        if self.stream is self.buffer_end:
            self._bufferup()
        if self.stream is NULL:
            # end of file
            self.c = 0
        else:
            self.c = self.stream[0]
            self.stream += 1

    cdef int read_line_into(self, CharVector* dest) except -1:  # nogil
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

    cdef int read_lines_into(self, CharVector* dest, Py_ssize_t n) except -1:
        """Read up to `n` lines into the `dest` buffer."""
        cdef Py_ssize_t n_lines_read = 0

        # with nogil:

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
        Py_ssize_t stream_index

    def __cinit__(self, Py_ssize_t capacity):
        CharVector_init(&self.vector, capacity)
        self.stream_index = 0

    def __dealloc__(self):
        CharVector_free(&self.vector)

    cdef int advance(self) except -1:  # nogil
        if self.stream_index < self.vector.size:
            self.c = self.vector.data[self.stream_index]
            self.stream_index += 1
        else:
            self.c = 0

    cdef void clear(self) :  # nogil
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
        list fields
        list formats

        # dynamic attributes - reflect current state during parsing
        int state  # overall parser state
        Py_ssize_t variant_index  # index of current variant
        Py_ssize_t chunk_variant_index  # index of current variant within current chunk
        Py_ssize_t sample_index  # index of current sample within call data
        Py_ssize_t sample_output_index  # index of current sample within output calldata arrays
        Py_ssize_t sample_field_index  # index of field within call data for current sample
        IntVector variant_format_indices  # indices of formats for the current variant

        # buffers
        CharVector temp  # used for numeric values
        CharVector info_key  # used for info key
        CharVector info_val  # used for info value

        # keep track of current chrom and pos, even if fields are skipped
        CharVector chrom
        long pos

        # track size of reference allele (needed for altlen)
        Py_ssize_t ref_len

    def __cinit__(self, headers, fields):
        self.headers = headers
        self.fields = list(fields)
        self.formats = list()
        for f in fields:
            group, name = f.split('/')
            if group == 'calldata':
                self.formats.append(name)

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


def check_samples(loc_samples, headers):
    n_samples = len(headers.samples)
    if loc_samples is None:
        loc_samples = np.ones(n_samples, dtype='u1')
    else:
        # assume samples is already a boolean indexing array
        loc_samples = loc_samples.view('u1')
        assert loc_samples.shape[0] == n_samples
    return loc_samples


cdef class VCFChunkIterator:

    cdef:
        InputStreamBase stream
        VCFContext context
        VCFParser parser

    def __init__(self,
                 InputStreamBase stream,
                 chunk_length,
                 headers,
                 fields,
                 types,
                 numbers,
                 fills,
                 region,
                 loc_samples):

        # store reference to input stream
        self.stream = stream

        # setup context
        fields = sorted(fields)
        self.context = VCFContext(headers, fields)

        # setup parser
        loc_samples = check_samples(loc_samples, headers)
        self.parser = VCFParser(fields=fields, types=types, numbers=numbers,
                                chunk_length=chunk_length, loc_samples=loc_samples,
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
        Py_ssize_t chunk_length
        np.uint8_t[:] loc_samples
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
        Py_ssize_t region_begin
        Py_ssize_t region_end

    def __init__(self, fields, types, numbers, chunk_length, loc_samples, fills, region):
        self.chunk_length = chunk_length
        self.loc_samples = loc_samples

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
            if PY2:
                self.region_chrom = tokens[0]
            else:
                self.region_chrom = tokens[0].encode('utf8')
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
            t = types[ID_FIELD]
            t = check_string_dtype(t)
            if t.kind == 'S':
                id_parser = VCFIDStringParser(dtype=t, chunk_length=self.chunk_length)
            else:
                id_parser = VCFIDObjectParser(chunk_length=self.chunk_length)
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
            t = check_string_dtype(t)
        if t is not None and t.kind == 'S':
            ref_parser = VCFRefStringParser(dtype=t, chunk_length=self.chunk_length, store=store)
        else:
            ref_parser = VCFRefObjectParser(chunk_length=self.chunk_length, store=store)
        ref_parser.malloc_chunk()
        self.ref_parser = ref_parser

    def _init_alt(self, fields, types, numbers):
        """Setup ALT parser."""

        store_alt = False
        store_numalt = False
        store_altlen = False
        store_is_snp = False
        t = types.get(ALT_FIELD, None)
        n = numbers.get(ALT_FIELD, 1)
        if ALT_FIELD in fields:
            store_alt = True
            fields.remove(ALT_FIELD)
        if NUMALT_FIELD in fields:
            store_numalt = True
            fields.remove(NUMALT_FIELD)
        if ALTLEN_FIELD in fields:
            store_altlen = True
            fields.remove(ALTLEN_FIELD)
        if IS_SNP_FIELD in fields:
            store_is_snp = True
            fields.remove(IS_SNP_FIELD)

        if store_alt or store_numalt or store_altlen or store_is_snp:
            if store_alt:
                t = check_string_dtype(t)
            if t is not None and t.kind == 'S':
                alt_parser = VCFAltStringParser(dtype=t, number=n, chunk_length=self.chunk_length,
                                                store_alt=store_alt, store_numalt=store_numalt,
                                                store_altlen=store_altlen, store_is_snp=store_is_snp)
            else:
                alt_parser = VCFAltObjectParser(number=n, chunk_length=self.chunk_length, store_alt=store_alt,
                                                store_numalt=store_numalt,
                                                store_altlen=store_altlen, store_is_snp=store_is_snp)
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
                k = field[16:]
                if isinstance(k, text_type):
                    k = k.encode('utf8')
                filter_keys.append(k)
                fields.remove(field)
        if filter_keys:
            filter_parser = VCFFilterParser(filter_keys=filter_keys, chunk_length=self.chunk_length)
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
                if isinstance(name, text_type):
                    key = name.encode('utf8')
                else:
                    key = name
                info_keys.append(key)
                fields.remove(field)
                info_types[key] = types[field]
                info_numbers[key] = numbers[field]
                if field in fills:
                    info_fills[key] = fills[field]
        if info_keys:
            info_parser = VCFInfoParser(info_keys=info_keys,
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
                if isinstance(name, text_type):
                    key = name.encode('utf8')
                else:
                    key = name
                format_keys.append(key)
                fields.remove(field)
                format_types[key] = types[field]
                format_numbers[key] = numbers[field]
                if field in fills:
                    format_fills[key] = fills[field]
        if format_keys:
            format_parser = VCFFormatParser(format_keys=format_keys)
            calldata_parser = VCFCallDataParser(format_keys=format_keys,
                                                types=format_types,
                                                numbers=format_numbers,
                                                chunk_length=self.chunk_length,
                                                loc_samples=self.loc_samples,
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

        # with nogil:

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
                    # with gil:
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
                # with gil:
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
        object dtype
        Py_ssize_t itemsize
        Py_ssize_t number
        object fill
        Py_ssize_t chunk_length
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        pass

    cdef int malloc_chunk(self) except -1:
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'variants/' + text_type(self.key, 'utf8')
        values = self.values
        if self.values.ndim > 1 and self.number == 1:
            values = values.squeeze(axis=1)
        chunk[field] = values[:limit]


cdef class VCFSkipFieldParser(VCFFieldParserBase):
    """Parser to skip a tab-delimited field."""

    def __init__(self, key):
        super(VCFSkipFieldParser, self).__init__(key=key)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil

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
    if dtype.kind not in ['S', 'O']:
        raise ValueError("expected string ('S') or object ('O') dtype, found: %r" % dtype)
    return dtype


cdef int vcf_read_field(InputStreamBase stream,
                        VCFContext context,
                        CharVector* dest) except -1:  # nogil

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
                           VCFContext context) except -1:  # nogil

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


cdef int vcf_skip_variant(InputStreamBase stream, VCFContext context) except -1:  # nogil
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
        Py_ssize_t region_begin
        Py_ssize_t region_end
        np.ndarray chrom_values
        np.ndarray pos_values

    def __init__(self, dtype, store_chrom, store_pos, chunk_length, region_chrom,
                 region_begin, region_end):
        if store_chrom:
            dtype = check_string_dtype(dtype)
        super(VCFChromPosParser, self).__init__(key=b'CHROM', dtype=dtype, number=1, chunk_length=chunk_length)
        self.store_chrom = store_chrom
        self.store_pos = store_pos
        if region_chrom:
            self.region_chrom = <char*> region_chrom
            self.region_begin = region_begin
            self.region_end = region_end
        else:
            self.region_chrom = NULL
            self.region_begin = 0
            self.region_end = 0

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t i, n, cmp
            # index into memory view
            Py_ssize_t memory_offset

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
            # with gil:
            warn('empty CHROM', context)
        CharVector_terminate(&context.chrom)

        # read pos
        if context.state == VCFState.CHROM:
            context.state += 1
            # read pos
            vcf_read_field(stream, context, &context.temp)
            if context.temp.size == 0:
                # with gil:
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
                vcf_skip_variant(stream, context)
                return 0

            if cmp > 0:
                vcf_skip_variant(stream, context)
                return 0

            if self.region_begin > 0 and context.pos < self.region_begin:
                vcf_skip_variant(stream, context)
                return 0

            if 0 < self.region_end < context.pos:
                vcf_skip_variant(stream, context)
                return 0

        # setup context
        context.sample_index = 0
        context.sample_field_index = 0
        context.sample_output_index = -1
        context.variant_index += 1
        context.chunk_variant_index += 1

        # store in chunk
        if self.store_chrom:

            if self.dtype.kind == 'S':

                # initialise memory index
                memory_offset = context.chunk_variant_index * self.itemsize

                # figure out how many characters to store
                n = min(context.chrom.size - 1, self.itemsize)

                # store characters
                for i in range(n):
                    self.chrom_memory[memory_offset + i] = context.chrom.data[i]

            else:
                # with gil:
                # N.B., don't include terminating null byte
                v = CharVector_to_pystr_sized(&context.chrom, context.chrom.size - 1)
                self.chrom_values[context.chunk_variant_index] = v

        if self.store_pos:
            self.pos_memory[context.chunk_variant_index] = context.pos

    cdef int malloc_chunk(self) except -1:
        if self.store_chrom:
            self.chrom_values = np.zeros(self.chunk_length, dtype=self.dtype)
            if self.dtype.kind == 'S':
                self.chrom_memory = self.chrom_values.view('u1')
            else:
                self.chrom_values.fill(u'')
        if self.store_pos:
            self.pos_values = np.zeros(self.chunk_length, dtype='int32')
            self.pos_memory = self.pos_values

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store_chrom:
            chunk[CHROM_FIELD] = self.chrom_values[:limit]
        if self.store_pos:
            chunk[POS_FIELD] = self.pos_values[:limit]


cdef class VCFIDStringParser(VCFFieldParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, dtype, chunk_length):
        super(VCFIDStringParser, self).__init__(key=b'ID', dtype=dtype, number=1, chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            # index into memory view
            Py_ssize_t memory_index = context.chunk_variant_index * self.itemsize
            # number of characters read into current value
            Py_ssize_t chars_stored = 0

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

    cdef int malloc_chunk(self) except -1:
        self.values = np.zeros(self.chunk_length, dtype=self.dtype)
        self.memory = self.values.view('u1')


cdef class VCFIDObjectParser(VCFFieldParserBase):

    def __init__(self, chunk_length):
        super(VCFIDObjectParser, self).__init__(key=b'ID', dtype=np.dtype('object'), number=1,
                                                chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil

        vcf_read_field(stream, context, &context.temp)

        # with gil:
        v = CharVector_to_pystr(&context.temp)
        self.values[context.chunk_variant_index] = v

        if context.state == VCFState.ID:
            context.state += 1

    cdef int malloc_chunk(self) except -1:
        self.values = np.empty(self.chunk_length, dtype=self.dtype)
        self.values.fill(u'')


cdef class VCFRefStringParser(VCFFieldParserBase):

    cdef:
        np.uint8_t[:] memory
        bint store

    def __init__(self, dtype, chunk_length, store):
        if store:
            dtype = check_string_dtype(dtype)
        super(VCFRefStringParser, self).__init__(key=b'REF', dtype=dtype, number=1, chunk_length=chunk_length)
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            # index into memory view
            Py_ssize_t memory_index = context.chunk_variant_index * self.itemsize
            # number of characters read into current value
            Py_ssize_t chars_stored = 0

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
                if stream.c != PERIOD:
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


cdef class VCFRefObjectParser(VCFFieldParserBase):

    cdef:
        bint store

    def __init__(self, chunk_length, store):
        super(VCFRefObjectParser, self).__init__(key=b'REF', dtype=np.dtype('object'), number=1, chunk_length=chunk_length)
        self.store = store

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil

        vcf_read_field(stream, context, &context.temp)

        # with gil:
        v = CharVector_to_pystr(&context.temp)
        if v != '.':
            context.ref_len = len(v)
        if self.store:
            self.values[context.chunk_variant_index] = v

        if context.state == VCFState.REF:
            context.state += 1

    cdef int malloc_chunk(self) except -1:
        if self.store:
            self.values = np.empty(self.chunk_length, dtype=self.dtype)
            self.values.fill(u'')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store:
            chunk[REF_FIELD] = self.values[:limit]


cdef class VCFAltStringParser(VCFFieldParserBase):

    cdef:
        np.uint8_t[:] memory
        np.int32_t[:] numalt_memory
        np.int32_t[:, :] altlen_memory
        np.uint8_t[:] is_snp_memory
        np.ndarray numalt_values
        np.ndarray altlen_values
        np.ndarray is_snp_values
        bint store_alt
        bint store_numalt
        bint store_altlen
        bint store_is_snp

    def __init__(self, dtype, number, chunk_length, store_alt, store_numalt, store_altlen,
                 store_is_snp):
        if store_alt:
            dtype = check_string_dtype(dtype)
        super(VCFAltStringParser, self).__init__(key=b'ALT', dtype=dtype, number=number,
                                                 chunk_length=chunk_length)
        self.store_alt = store_alt
        self.store_numalt = store_numalt
        self.store_altlen = store_altlen
        self.store_is_snp = store_is_snp

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            # index of alt values
            Py_ssize_t alt_index = 0
            # index into memory view
            Py_ssize_t memory_offset, memory_index
            # number of characters read into current value
            Py_ssize_t chars_stored = 0
            # size of alt allel
            Py_ssize_t alt_len = 0
            # is the variant a SNP?
            bint is_snp = context.ref_len == 1

        # bail out early for missing value
        if stream.c == PERIOD:
            # treat period as missing value, regardless of what comes next
            vcf_parse_missing(stream, context)
            return 0

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
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                context.state = VCFState.EOL
                break

            if stream.c == TAB:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                stream.advance()
                context.state += 1
                break

            elif stream.c == COMMA:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                # advance value index
                alt_index += 1
                # reset alt_len
                alt_len = 0
                # set memory index to beginning of next item
                memory_index = memory_offset + (alt_index * self.itemsize)
                # reset chars stored
                chars_stored = 0

            else:
                if stream.c != PERIOD and stream.c != ASTERISK:
                    alt_len += 1
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
        if self.store_is_snp:
            self.is_snp_memory[context.chunk_variant_index] = is_snp

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        if self.store_alt:
            self.values = np.zeros(shape, dtype=self.dtype, order='C')
            self.memory = self.values.reshape(-1).view('u1')
        if self.store_numalt:
            self.numalt_values = np.zeros(self.chunk_length, dtype='int32')
            self.numalt_memory = self.numalt_values
        if self.store_altlen:
            self.altlen_values = np.zeros(shape, dtype='int32')
            self.altlen_memory = self.altlen_values
        if self.store_is_snp:
            self.is_snp_values = np.zeros(self.chunk_length, dtype=bool)
            self.is_snp_memory = self.is_snp_values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store_alt:
            field = 'variants/' + text_type(self.key, 'utf8')
            values = self.values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]
        if self.store_numalt:
            field = NUMALT_FIELD
            values = self.numalt_values
            chunk[field] = values[:limit]
        if self.store_altlen:
            field = ALTLEN_FIELD
            values = self.altlen_values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]
        if self.store_is_snp:
            field = IS_SNP_FIELD
            values = self.is_snp_values
            chunk[field] = values[:limit]


cdef class VCFAltObjectParser(VCFFieldParserBase):

    cdef:
        np.int32_t[:] numalt_memory
        np.int32_t[:, :] altlen_memory
        np.uint8_t[:] is_snp_memory
        np.ndarray numalt_values
        np.ndarray altlen_values
        np.ndarray is_snp_values
        bint store_alt
        bint store_numalt
        bint store_altlen
        bint store_is_snp

    def __init__(self, number, chunk_length, store_alt, store_numalt, store_altlen, store_is_snp):
        super(VCFAltObjectParser, self).__init__(key=b'ALT', dtype=np.dtype('object'), number=number,
                                                 chunk_length=chunk_length)
        self.store_alt = store_alt
        self.store_numalt = store_numalt
        self.store_altlen = store_altlen
        self.store_is_snp = store_is_snp

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            # index of alt values
            Py_ssize_t alt_index = 0
            # size of alt allele
            Py_ssize_t alt_len = 0
            # is the variant a SNP?
            bint is_snp = context.ref_len == 1

        # bail out early for missing value
        if stream.c == PERIOD:
            # treat period as missing value, regardless of what comes next
            vcf_parse_missing(stream, context)
            return 0

        # bail out early for empty value
        if stream.c == TAB:
            stream.advance()
            context.state += 1
            return 0

        # setup temp
        CharVector_clear(&context.temp)

        while True:

            if stream.c == 0:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                if self.store_alt and alt_index < self.number and context.temp.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, alt_index] = v
                context.state = VCFState.EOF
                break

            elif stream.c == LF or stream.c == CR:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                if self.store_alt and alt_index < self.number and context.temp.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, alt_index] = v
                context.state = VCFState.EOL
                break

            if stream.c == TAB:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                if self.store_alt and alt_index < self.number and context.temp.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, alt_index] = v
                stream.advance()
                context.state += 1
                break

            elif stream.c == COMMA:
                is_snp = is_snp and alt_len == 1
                if self.store_altlen and alt_index < self.number:
                    self.altlen_memory[context.chunk_variant_index, alt_index] = \
                        alt_len - context.ref_len
                if self.store_alt and alt_index < self.number:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, alt_index] = v
                # advance value index
                alt_index += 1
                # reset
                CharVector_clear(&context.temp)
                alt_len = 0

            else:
                if stream.c != PERIOD and stream.c != ASTERISK:
                    alt_len += 1
                if self.store_alt and alt_index < self.number:
                    CharVector_append(&context.temp, stream.c)

            # advance input stream
            stream.advance()

        if self.store_numalt:
            self.numalt_memory[context.chunk_variant_index] = alt_index + 1
        if self.store_is_snp:
            self.is_snp_memory[context.chunk_variant_index] = is_snp

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        if self.store_alt:
            self.values = np.empty(shape, dtype=self.dtype, order='C')
            self.values.fill(u'')
        if self.store_numalt:
            self.numalt_values = np.zeros(self.chunk_length, dtype='int32')
            self.numalt_memory = self.numalt_values
        if self.store_altlen:
            self.altlen_values = np.zeros(shape, dtype='int32')
            self.altlen_memory = self.altlen_values
        if self.store_is_snp:
            self.is_snp_values = np.zeros(self.chunk_length, dtype=bool)
            self.is_snp_memory = self.is_snp_values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        if self.store_alt:
            field = 'variants/' + text_type(self.key, 'utf8')
            values = self.values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]
        if self.store_numalt:
            field = NUMALT_FIELD
            values = self.numalt_values
            chunk[field] = values[:limit]
        if self.store_altlen:
            field = ALTLEN_FIELD
            values = self.altlen_values
            if self.values.ndim > 1 and self.number == 1:
                values = values.squeeze(axis=1)
            chunk[field] = values[:limit]
        if self.store_is_snp:
            field = IS_SNP_FIELD
            values = self.is_snp_values
            chunk[field] = values[:limit]


cdef class VCFQualParser(VCFFieldParserBase):

    cdef np.float32_t[:] memory

    def __init__(self, chunk_length, fill=NAN):
        super(VCFQualParser, self).__init__(key=b'QUAL', dtype='float32', number=1,
                                            fill=fill, chunk_length=chunk_length)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            double value
            Py_ssize_t parsed

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
        tuple filter_keys
        Py_ssize_t n_filters
        char** filter_keys_cstr

    def __cinit__(self, filter_keys, chunk_length):
        cdef:
            Py_ssize_t i
        # N.B., need to keep a reference to these, otherwise C strings will not behave
        self.filter_keys = tuple(sorted(filter_keys))
        self.n_filters = len(self.filter_keys)
        self.filter_keys_cstr = <char**> malloc(sizeof(char*) * self.n_filters)
        for i in range(self.n_filters):
            self.filter_keys_cstr[i] = <char*> self.filter_keys[i]

    def __init__(self, filter_keys, chunk_length):
        super(VCFFilterParser, self).__init__(key=b'FILTER', dtype='bool', number=len(filter_keys), fill=0,
                                              chunk_length=chunk_length)

    def __dealloc__(self):
        if self.filter_keys_cstr is not NULL:
            free(self.filter_keys_cstr)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t filter_index

        # check for explicit missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            vcf_parse_missing(stream, context)
            return 0

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

    cdef int parse_filter(self, VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t filter_index
            Py_ssize_t i

        if context.temp.size == 0:
            # not strictly kosher, treat as missing/empty
            return 0

        CharVector_terminate(&context.temp)

        # search through filters to find index
        filter_index = search_sorted_cstr(context.temp.data, self.filter_keys_cstr, self.n_filters)

        # store value
        if filter_index >= 0:
            self.memory[context.chunk_variant_index, filter_index] = 1

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_filters)
        self.values = np.zeros(shape, dtype=bool)
        self.memory = self.values.view('u1')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        for i, f in enumerate(self.filter_keys):
            f = text_type(f, 'utf8')
            field = 'variants/FILTER_' + f
            chunk[field] = self.values[:limit, i]


##########################################################################################
# INFO parsing


cdef class VCFInfoParser(VCFFieldParserBase):

    cdef:
        tuple info_keys
        Py_ssize_t n_infos
        char** info_keys_cstr
        tuple info_parsers
        PyObject** info_parsers_cptr
        VCFInfoParserBase skip_parser
        object fills

    def __cinit__(self, info_keys, types, numbers, chunk_length, fills):

        # setup INFO keys
        # N.B., need to keep a reference to these, otherwise C strings will not behave
        self.info_keys = tuple(sorted(info_keys))
        self.n_infos = len(self.info_keys)

        # setup INFO keys as C strings for nogil searching
        self.info_keys_cstr = <char**> malloc(sizeof(char*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_keys_cstr[i] = <char*> self.info_keys[i]

        # setup INFO parsers
        info_parsers = list()
        self.skip_parser = VCFInfoSkipParser(key=None)
        for key in self.info_keys:
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
            elif t.kind == 'O':
                parser = VCFInfoObjectParser(key, number=n, chunk_length=chunk_length)
            else:
                parser = VCFInfoSkipParser(key)
                warnings.warn('type %s not supported for INFO field %r, field will be skipped' % (t, key))
            info_parsers.append(parser)
        self.info_parsers = tuple(info_parsers)

        # store pointers to parsers for nogil trickery
        self.info_parsers_cptr = <PyObject**> malloc(sizeof(PyObject*) * self.n_infos)
        for i in range(self.n_infos):
            self.info_parsers_cptr[i] = <PyObject*> self.info_parsers[i]

    def __init__(self, info_keys, types, numbers, chunk_length, fills):
        super(VCFInfoParser, self).__init__(key=b'INFO', chunk_length=chunk_length)
        self.fills = fills

    def __dealloc__(self):
        if self.info_keys_cstr is not NULL:
            free(self.info_keys_cstr)
        if self.info_parsers_cptr is not NULL:
            free(self.info_parsers_cptr)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil

        # reset buffers
        CharVector_clear(&context.info_key)
        CharVector_clear(&context.info_val)

        # check for missing value
        if stream.c == PERIOD:
            # treat leading period as missing, regardless of what comes next
            vcf_parse_missing(stream, context)
            return 0

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
                    # with gil:
                    warn('missing INFO key', context)
                    self.skip_parser.parse(stream, context)

            else:

                CharVector_append(&context.info_key, stream.c)
                stream.advance()

    cdef int parse_info(self,
                        InputStreamBase stream,
                        VCFContext context) except -1:  # nogil

        cdef:
            Py_ssize_t parser_index
            PyObject* parser

        # terminate key
        CharVector_terminate(&context.info_key)

        # search for index of current INFO key
        parser_index = search_sorted_cstr(context.info_key.data, self.info_keys_cstr, self.n_infos)

        # clear out key for good measure
        CharVector_clear(&context.info_key)

        if parser_index >= 0:
            # obtain parser, use trickery for nogil
            parser = self.info_parsers_cptr[parser_index]
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
        object dtype
        Py_ssize_t itemsize
        Py_ssize_t number
        object fill
        np.ndarray values
        Py_ssize_t chunk_length

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        pass

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'variants/' + text_type(self.key, 'utf8')
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_floating(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_info_parse_floating(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
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
        field = 'variants/' + text_type(self.key, 'utf8')
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t value_index = 0
            # index into memory view
            Py_ssize_t memory_offset, memory_index
            # number of characters read into current value
            Py_ssize_t chars_stored = 0

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


cdef class VCFInfoObjectParser(VCFInfoParserBase):

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = np.dtype('object')
        super(VCFInfoObjectParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t value_index = 0

        CharVector_clear(&context.info_val)

        while True:

            if stream.c == 0 or \
                    stream.c == LF or \
                    stream.c == CR or \
                    stream.c == TAB or \
                    stream.c == SEMICOLON:
                if value_index < self.number and context.info_val.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.info_val)
                    self.values[context.chunk_variant_index, value_index] = v
                break

            elif stream.c == COMMA:
                if value_index < self.number and context.info_val.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.info_val)
                    self.values[context.chunk_variant_index, value_index] = v
                    CharVector_clear(&context.info_val)
                # advance value index
                value_index += 1

            elif value_index < self.number:
                CharVector_append(&context.info_val, stream.c)

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.values.fill(u'')


cdef class VCFInfoSkipParser(VCFInfoParserBase):

    def __init__(self, *args, **kwargs):
        super(VCFInfoSkipParser, self).__init__(*args, **kwargs)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
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
                                integer[:, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t value_index = 0

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
                                Py_ssize_t value_index,
                                integer[:, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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
                                 floating[:, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t value_index = 0

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
                                 Py_ssize_t value_index,
                                 floating[:, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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
        tuple format_keys
        Py_ssize_t n_formats
        char** format_keys_cstr

    def __cinit__(self, format_keys):

        # setup FORMAT keys
        # N.B., need to keep a reference to these, otherwise C strings will not behave
        self.format_keys = tuple(sorted(format_keys))
        self.n_formats = len(self.format_keys)

        # setup FORMAT keys as C strings for nogil searching
        self.format_keys_cstr = <char**> malloc(sizeof(char*) * self.n_formats)
        for i in range(self.n_formats):
            self.format_keys_cstr[i] = <char*> self.format_keys[i]

    def __init__(self, format_keys):
        super(VCFFormatParser, self).__init__(key=b'FORMAT')

    def __dealloc__(self):
        if self.format_keys_cstr is not NULL:
            free(self.format_keys_cstr)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil

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

    cdef int store_format(self, VCFContext context) except -1:  # nogil
        cdef Py_ssize_t format_index

        # deal with empty or missing data
        if context.temp.size == 0:
            # not strictly kosher, handle as missing/empty
            return 0

        if context.temp.size == 1 and context.temp.data[0] == PERIOD:
            return 0

        # terminate the string
        CharVector_terminate(&context.temp)

        # find format index
        format_index = search_sorted_cstr(context.temp.data, self.format_keys_cstr, self.n_formats)

        # add to vector of indices for the current variant
        IntVector_append(&context.variant_format_indices, format_index)

        # clear out temp
        CharVector_clear(&context.temp)



cdef class VCFSkipAllCallDataParser(VCFFieldParserBase):
    """Skip a field."""

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_skip_variant(stream, context)

    cdef int make_chunk(self, chunk, limit=None) except -1:
        pass


cdef class VCFCallDataParser(VCFFieldParserBase):

    cdef:
        tuple format_keys
        Py_ssize_t n_formats
        tuple parsers
        PyObject** parsers_cptr
        VCFCallDataParserBase skip_parser
        np.uint8_t[:] loc_samples
        Py_ssize_t n_samples
        Py_ssize_t n_samples_out

    def __cinit__(self, format_keys, types, numbers, chunk_length, loc_samples, fills):
        self.chunk_length = chunk_length
        self.loc_samples = loc_samples
        self.n_samples = loc_samples.shape[0]
        self.n_samples_out = np.count_nonzero(loc_samples)

        # setup formats
        self.format_keys = tuple(sorted(format_keys))
        self.n_formats = len(self.format_keys)

        # setup parsers
        self.skip_parser = VCFCallDataSkipParser(key=None)
        parsers = list()
        kwds = dict(chunk_length=chunk_length, n_samples_out=self.n_samples_out)
        for key in self.format_keys:
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
            elif t.kind == 'O':
                parser = VCFCallDataObjectParser(key, number=n, **kwds)

            else:
                parser = VCFCallDataSkipParser(key)
                warnings.warn('type %r not supported for FORMAT field %r, field will be skipped' % (t, key))

            parsers.append(parser)
        self.parsers = tuple(parsers)

        # store pointers to parsers
        self.parsers_cptr = <PyObject**> malloc(sizeof(PyObject*) * self.n_formats)
        for i in range(self.n_formats):
            self.parsers_cptr[i] = <PyObject*> self.parsers[i]

    def __init__(self, format_keys, types, numbers, chunk_length, loc_samples, fills):
        super(VCFCallDataParser, self).__init__(chunk_length=chunk_length)

    def __dealloc__(self):
        if self.parsers_cptr is not NULL:
            free(self.parsers_cptr)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        cdef:
            PyObject* parser

        # initialise context
        context.sample_index = 0
        context.sample_output_index = -1
        context.sample_field_index = 0

        # setup output indexing
        if self.loc_samples[0]:
            context.sample_output_index += 1
        else:
            # skip to next sample
            while stream.c != 0 and stream.c != LF and stream.c != CR and stream.c != TAB:
                stream.advance()

        while True:

            if context.sample_index >= self.n_samples:
                # with gil:
                warn('more samples than given in header', context)
                while stream.c != 0 and stream.c != LF and stream.c != CR:
                    stream.advance()

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
                if self.loc_samples[context.sample_index]:
                    context.sample_output_index += 1
                else:
                    # skip to next sample
                    while stream.c != 0 and stream.c != LF and stream.c != CR and stream.c != TAB:
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
                    parser = self.parsers_cptr[format_index]
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
        object dtype
        Py_ssize_t itemsize
        Py_ssize_t number
        object fill
        np.ndarray values
        Py_ssize_t chunk_length
        Py_ssize_t n_samples_out

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
        field = 'calldata/' + text_type(self.key, 'utf8')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        pass


cdef class VCFCallDataSkipParser(VCFCallDataParserBase):

    def __init__(self, key, *args, **kwargs):
        super(VCFCallDataSkipParser, self).__init__(key=key)

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_genotype_parse(InputStreamBase stream,
                            VCFContext context,
                            integer[:, :, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t value_index = 0

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
                            Py_ssize_t value_index) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_genotype_ac_parse(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = 0


cdef int vcf_genotype_ac_parse(InputStreamBase stream,
                               VCFContext context,
                               integer[:, :, :] memory) except -1:  # nogil
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
                               integer[:, :, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_integer(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_floating(stream, context, self.memory)

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

    cdef int parse(self, InputStreamBase stream, VCFContext context) except -1:  # nogil
        vcf_calldata_parse_floating(stream, context, self.memory)

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.memory = self.values
        self.memory[:] = self.fill


cdef int vcf_calldata_parse_integer(InputStreamBase stream,
                                    VCFContext context,
                                    integer[:, :, :] memory) except -1:  # nogil

    cdef:
        Py_ssize_t value_index = 0

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
                                    Py_ssize_t value_index,
                                    integer[:, :, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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
                                     floating[:, :, :] memory) except -1:  # nogil

    cdef:
        Py_ssize_t value_index = 0

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
                                     Py_ssize_t value_index,
                                     floating[:, :, :] memory) except -1:  # nogil
    cdef:
        Py_ssize_t parsed
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
                   VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t value_index = 0
            # index into memory view
            Py_ssize_t memory_offset, memory_index
            # number of characters read into current value
            Py_ssize_t chars_stored = 0

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
        field = 'calldata/' + text_type(self.key, 'utf8')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values


cdef class VCFCallDataObjectParser(VCFCallDataParserBase):

    cdef np.uint8_t[:] memory

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = np.dtype('object')
        super(VCFCallDataObjectParser, self).__init__(*args, **kwargs)

    cdef int parse(self,
                   InputStreamBase stream,
                   VCFContext context) except -1:  # nogil
        cdef:
            Py_ssize_t value_index = 0

        CharVector_clear(&context.temp)

        # read characters until tab
        while True:

            if stream.c == TAB or \
                    stream.c == COLON or \
                    stream.c == CR or \
                    stream.c == LF or \
                    stream.c == 0:
                if value_index < self.number and context.temp.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, context.sample_output_index, value_index] = v
                break

            elif stream.c == COMMA:
                if value_index < self.number and context.temp.size > 0:
                    # with gil:
                    v = CharVector_to_pystr(&context.temp)
                    self.values[context.chunk_variant_index, context.sample_output_index, value_index] = v
                CharVector_clear(&context.temp)
                # advance value index
                value_index += 1

            elif value_index < self.number:
                CharVector_append(&context.temp, stream.c)

            # advance input stream
            stream.advance()

    cdef int malloc_chunk(self) except -1:
        shape = (self.chunk_length, self.n_samples_out, self.number)
        self.values = np.empty(shape, dtype=self.dtype)
        self.values.fill(u'')

    cdef int make_chunk(self, chunk, limit=None) except -1:
        field = 'calldata/' + text_type(self.key, 'utf8')
        values = self.values[:limit]
        if self.number == 1:
            values = values.squeeze(axis=2)
        chunk[field] = values


##########################################################################################
# Low-level VCF value parsing functions


cdef Py_ssize_t vcf_strtol(CharVector* value, VCFContext context, long* l) except -1:  # nogil
    cdef:
        char* str_end
        Py_ssize_t parsed

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
        # with gil:
        warn('not all characters parsed for integer value', context)
        return parsed

    else:
        # with gil:
        warn('error parsing integer value', context)
        return 0


cdef Py_ssize_t vcf_strtod(CharVector* value, VCFContext context, double* d) except -1:  # nogil
    cdef:
        char* str_end
        Py_ssize_t parsed

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
        # with gil:
        warn('not all characters parsed for floating point value', context)
        return parsed

    else:
        # with gil:
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


cdef int warn(message, VCFContext context) except -1:
    cdef Py_ssize_t format_index
    message += '; field: %s' % vcf_state_labels[context.state]
    message += '; variant: %s' % context.variant_index
    if context.state > VCFState.POS:
        chrom = CharVector_to_pybytes(&context.chrom)
        if not PY2:
            chrom = str(chrom, 'utf8')
        message += ' (%s:%s)' % (chrom, context.pos)
    if context.state == VCFState.CALLDATA:
        if context.sample_index >= len(context.headers.samples):
            sample = 'unknown'
        else:
            sample = context.headers.samples[context.sample_index]
        if context.sample_field_index >= context.variant_format_indices.size:
            format = 'unknown'
        else:
            format_index = context.variant_format_indices.data[context.sample_field_index]
            format = context.formats[format_index]
        message += '; sample: %s:%s (%s:%s)' % (context.sample_index,
                                                context.sample_field_index,
                                                sample,
                                                format)
    warnings.warn(message)


# import sys
#
#
# cdef int debug(message, vars=None) except -1:
#     message = '[DEBUG] ' + str(message)
#     if vars:
#         message = message % vars
#     print(message, file=sys.stderr)
#     sys.stderr.flush()


##########################################################################################
# EXPERIMENTAL support for multi-threaded parsing
# N.B., this is not used for the moment, because use of object dtype for strings
# requires GIL acquisition, and this may hurt performance in a single-threaded
# context. I'm not completely certain that is the case, but I am out of time to
# explore further.


# import itertools
# import time
#
#
# cdef class VCFParallelParser:
#
#     cdef:
#         FileInputStream stream
#         CharVectorInputStream buffer
#         VCFContext context
#         VCFParser parser
#         Py_ssize_t chunk_length
#         Py_ssize_t block_length
#         object pool
#         object result
#
#     def __cinit__(self, stream, parser, chunk_length, block_length, pool, headers, fields):
#         self.buffer = CharVectorInputStream(2**14)
#         self.context = VCFContext(headers, fields)
#         self.stream = stream
#         self.parser = parser
#         self.chunk_length = chunk_length
#         self.block_length = block_length
#         self.pool = pool
#         self.result = None
#
#     def read(self, n_lines):
#         self.buffer.clear()
#         n_lines_read = self.stream.read_lines_into(&(self.buffer.vector), n_lines)
#         self.buffer.advance()
#         return n_lines_read
#
#     def parse_async(self, block_index, chunk_index):
#         self.result = self.pool.apply_async(self.parse, args=(block_index, chunk_index))
#
#     def join(self):
#         if self.result is not None:
#             self.result.get()
#
#     def parse(self, block_index, chunk_index):
#         before = time.time()
#         # set initial state
#         self.context.state = VCFState.CHROM
#         self.context.chunk_variant_index = block_index * self.block_length - 1
#         self.context.variant_index = (chunk_index * self.chunk_length +
#                                       self.context.chunk_variant_index)
#         # parse the block of data stored in the buffer
#         self.parser.parse(self.buffer, self.context)
#         after = time.time()
#
#
# cdef class VCFParallelChunkIterator:
#
#     cdef:
#         FileInputStream stream
#         VCFParser parser
#         object pool
#         Py_ssize_t chunk_length
#         Py_ssize_t block_length
#         int n_threads
#         int n_workers
#         Py_ssize_t chunk_index
#         list workers
#
#     def __cinit__(self,
#                   FileInputStream stream,
#                   Py_ssize_t chunk_length, Py_ssize_t block_length, int n_threads,
#                   headers, fields, types, numbers, fills, region, loc_samples):
#
#         fields = sorted(fields)
#         self.stream = stream
#         self.chunk_length = chunk_length
#         self.n_threads = n_threads
#         self.pool = ThreadPool(n_threads)
#         # allow one more worker than number of threads in pool to allow for sync
#         # reading of data in the main thread
#         self.n_workers = n_threads + 1
#         # only makes sense to have block length at most fraction chunk length if we want
#         # some parallelism
#         self.block_length = min(block_length, chunk_length//self.n_workers)
#         if self.block_length < 1:
#             self.block_length = 1
#         loc_samples = check_samples(loc_samples, headers)
#         self.parser = VCFParser(
#             fields=fields, types=types, numbers=numbers, chunk_length=chunk_length, loc_samples=loc_samples, fills=fills,
#             region=region
#         )
#         self.chunk_index = -1
#         self.workers = [VCFParallelParser(stream=stream, parser=self.parser, chunk_length=self.chunk_length,
#                                           block_length=self.block_length, pool=self.pool, headers=headers,
#                                           fields=fields)
#                         for _ in range(self.n_workers)]
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         cdef:
#             Py_ssize_t block_index = 0
#             Py_ssize_t i = 0
#             Py_ssize_t n_lines
#             Py_ssize_t n_lines_read = 0
#             VCFParallelParser worker
#
#         # increment the current chunk index
#         self.chunk_index += 1
#
#         # allocate arrays for next chunk
#         self.parser.malloc_chunk()
#
#         # cycle around the workers
#         for i in itertools.cycle(list(range(self.n_workers))):
#             worker = self.workers[i]
#
#             # wait for the result to finish - this ensures we don't overwrite a
#             # worker's buffer while it's still parsing
#             worker.join()
#
#             # read lines into the worker's buffer - this part has to be synchronous
#             n_lines = min(self.block_length, self.chunk_length - n_lines_read)
#             n_lines_read += worker.read(n_lines)
#
#             # launch parsing of the block in parallel
#             worker.parse_async(block_index, self.chunk_index)
#
#             # increment the current block index
#             block_index += 1
#
#             # is the chunk done?
#             if n_lines_read >= self.chunk_length:
#                 break
#
#             # is the input stream exhausted?
#             if self.stream.c == 0:
#                 break
#
#         # wait for all parallel tasks to complete
#         for worker in self.workers:
#             worker.join()
#
#         # obtain the final chunk length via the last worker
#         worker = self.workers[i]
#         chunk_length = worker.context.chunk_variant_index + 1
#
#         # obtain the chunk
#         chunk = self.parser.make_chunk(chunk_length)
#
#         if chunk is None:
#             # clean up thread pool
#             self.pool.close()
#             self.pool.join()
#             self.pool.terminate()
#             raise StopIteration
#
#         else:
#             chrom = CharVector_to_pybytes(&worker.context.chrom)
#             pos = worker.context.pos
#             return chunk, chunk_length, chrom, pos


###################################################################
# ANN transformer


# ANN field indices
cdef enum ANNFidx:
    ALLELE = 0,
    ANNOTATION = 1,
    ANNOTATION_IMPACT = 2,
    GENE_NAME = 3,
    GENE_ID = 4,
    FEATURE_TYPE = 5,
    FEATURE_ID = 6,
    TRANSCRIPT_BIOTYPE = 7,
    RANK = 8,
    HGVS_C = 9,
    HGVS_P = 10,
    CDNA = 11,
    CDS = 12,
    AA = 13,
    DISTANCE = 14


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
ANN_CDNA_POS_FIELD = 'variants/ANN_cDNA_pos'
ANN_CDNA_LENGTH_FIELD = 'variants/ANN_cDNA_length'
ANN_CDS_FIELD = 'variants/ANN_CDS'
ANN_CDS_POS_FIELD = 'variants/ANN_CDS_pos'
ANN_CDS_LENGTH_FIELD = 'variants/ANN_CDS_length'
ANN_AA_FIELD = 'variants/ANN_AA'
ANN_AA_POS_FIELD = 'variants/ANN_AA_pos'
ANN_AA_LENGTH_FIELD = 'variants/ANN_AA_length'
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
    ANN_CDNA_POS_FIELD,
    ANN_CDNA_LENGTH_FIELD,
    ANN_CDS_POS_FIELD,
    ANN_CDS_LENGTH_FIELD,
    ANN_AA_POS_FIELD,
    ANN_AA_LENGTH_FIELD,
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
    return f


def _normalize_ann_fields(fields):
    normed_fields = list()

    if fields is None:
        return list(ANN_FIELDS)

    else:
        for f in fields:
            f = _normalize_ann_field_prefix(f)
            # convenience features
            if f == ANN_CDNA_FIELD:
                for ff in ANN_CDNA_POS_FIELD, ANN_CDNA_LENGTH_FIELD:
                    if ff not in normed_fields:
                        normed_fields.append(ff)
            elif f == ANN_CDS_FIELD:
                for ff in ANN_CDS_POS_FIELD, ANN_CDS_LENGTH_FIELD:
                    if ff not in normed_fields:
                        normed_fields.append(ff)
            elif f == ANN_AA_FIELD:
                for ff in ANN_AA_POS_FIELD, ANN_AA_LENGTH_FIELD:
                    if ff not in normed_fields:
                        normed_fields.append(ff)
            # all other fields
            elif f is not None and f not in normed_fields:
                if f not in ANN_FIELDS:
                    warnings.warn('invalid ANN field %r, will be ignored' % f)
                else:
                    normed_fields.append(f)

    return normed_fields


default_ann_types = dict()
default_ann_types[ANN_ALLELE_FIELD] = np.dtype('object')
default_ann_types[ANN_ANNOTATION_FIELD] = np.dtype('object')
default_ann_types[ANN_ANNOTATION_IMPACT_FIELD] = np.dtype('object')
default_ann_types[ANN_GENE_NAME_FIELD] = np.dtype('object')
default_ann_types[ANN_GENE_ID_FIELD] = np.dtype('object')
default_ann_types[ANN_FEATURE_TYPE_FIELD] = np.dtype('object')
default_ann_types[ANN_FEATURE_ID_FIELD] = np.dtype('object')
default_ann_types[ANN_TRANSCRIPT_BIOTYPE_FIELD] = np.dtype('object')
default_ann_types[ANN_RANK_FIELD] = np.dtype('int8')
default_ann_types[ANN_HGVS_C_FIELD] = np.dtype('object')
default_ann_types[ANN_HGVS_P_FIELD] = np.dtype('object')
default_ann_types[ANN_CDNA_POS_FIELD] = np.dtype('int32')
default_ann_types[ANN_CDNA_LENGTH_FIELD] = np.dtype('int32')
default_ann_types[ANN_CDS_POS_FIELD] = np.dtype('int32')
default_ann_types[ANN_CDS_LENGTH_FIELD] = np.dtype('int32')
default_ann_types[ANN_AA_POS_FIELD] = np.dtype('int32')
default_ann_types[ANN_AA_LENGTH_FIELD] = np.dtype('int32')
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
        list fields
        object types
        bint keep_original

    def __init__(self, fields=None, types=None, keep_original=False):
        self.fields = _normalize_ann_fields(fields)
        self.types = _normalize_ann_types(self.fields, types)
        self.keep_original = keep_original

    def transform_fields(self, fields):
        fields_transformed = list()
        for f in fields:
            if f == ANN_FIELD:
                if self.keep_original:
                    fields_transformed.append(f)
                fields_transformed.extend(self.fields)
            else:
                fields_transformed.append(f)
        return fields_transformed

    def _malloc_string_array(self, field, shape):
        if field in self.fields:
            t = check_string_dtype(self.types[field])
            a = np.empty(shape, dtype=t)
            if t.kind == 'S':
                a.fill(b'')
            else:
                a.fill(u'')
        else:
            a = None
        return a

    def _malloc_integer_array(self, field, shape):
        if field in self.fields:
            t = self.types[field]
            if t.kind != 'i':
                raise ValueError('only signed integer dtype supported for field %r' % field)
            a = np.empty(shape, dtype=t)
            a.fill(-1)
        else:
            a = None
        return a

    def transform_chunk(self, chunk):
        cdef:
            Py_ssize_t i, j, chunk_length, number, n_vals
            list vals
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
        allele = self._malloc_string_array(ANN_ALLELE_FIELD, shape)
        annotation = self._malloc_string_array(ANN_ANNOTATION_FIELD, shape)
        annotation_impact = self._malloc_string_array(ANN_ANNOTATION_IMPACT_FIELD, shape)
        gene_name = self._malloc_string_array(ANN_GENE_NAME_FIELD, shape)
        gene_id = self._malloc_string_array(ANN_GENE_ID_FIELD, shape)
        feature_type = self._malloc_string_array(ANN_FEATURE_TYPE_FIELD, shape)
        feature_id = self._malloc_string_array(ANN_FEATURE_ID_FIELD, shape)
        transcript_biotype = self._malloc_string_array(ANN_TRANSCRIPT_BIOTYPE_FIELD, shape)
        rank = self._malloc_integer_array(ANN_RANK_FIELD, shape)
        hgvs_c = self._malloc_string_array(ANN_HGVS_C_FIELD, shape)
        hgvs_p = self._malloc_string_array(ANN_HGVS_P_FIELD, shape)
        cdna_pos = self._malloc_integer_array(ANN_CDNA_POS_FIELD, shape)
        cdna_length = self._malloc_integer_array(ANN_CDNA_LENGTH_FIELD, shape)
        cds_pos = self._malloc_integer_array(ANN_CDS_POS_FIELD, shape)
        cds_length = self._malloc_integer_array(ANN_CDS_LENGTH_FIELD, shape)
        aa_pos = self._malloc_integer_array(ANN_AA_POS_FIELD, shape)
        aa_length = self._malloc_integer_array(ANN_AA_LENGTH_FIELD, shape)
        distance = self._malloc_integer_array(ANN_DISTANCE_FIELD, shape)

        # start working
        for i in range(chunk_length):
            for j in range(number):

                # obtain raw string value
                raw = ann[i, j]
                if not PY2 and isinstance(raw, bytes):
                    raw = str(raw, 'utf8')

                # bail early if no content
                if raw == '' or raw == '.':
                    continue

                # split fields
                vals = raw.split('|')
                n_vals = len(vals)

                # convert and store values
                if allele is not None and n_vals > ANNFidx.ALLELE:
                    allele[i, j] = vals[ANNFidx.ALLELE]
                if annotation is not None and n_vals > ANNFidx.ANNOTATION:
                    annotation[i, j] = vals[ANNFidx.ANNOTATION]
                if annotation_impact is not None and n_vals > ANNFidx.ANNOTATION_IMPACT:
                    annotation_impact[i, j] = vals[ANNFidx.ANNOTATION_IMPACT]
                if gene_name is not None and n_vals > ANNFidx.GENE_NAME:
                    gene_name[i, j] = vals[ANNFidx.GENE_NAME]
                if gene_id is not None and n_vals > ANNFidx.GENE_ID:
                    gene_id[i, j] = vals[ANNFidx.GENE_ID]
                if feature_type is not None and n_vals > ANNFidx.FEATURE_TYPE:
                    feature_type[i, j] = vals[ANNFidx.FEATURE_TYPE]
                if feature_id is not None and n_vals > ANNFidx.FEATURE_ID:
                    feature_id[i, j] = vals[ANNFidx.FEATURE_ID]
                if transcript_biotype is not None and n_vals > ANNFidx.TRANSCRIPT_BIOTYPE:
                    transcript_biotype[i, j] = vals[ANNFidx.TRANSCRIPT_BIOTYPE]
                if rank is not None and n_vals > ANNFidx.RANK:
                    v = vals[ANNFidx.RANK]
                    if v:
                        vv = v.split('/')
                        # ignore second part of rank
                        rank[i, j] = int(vv[0])
                if hgvs_c is not None and n_vals > ANNFidx.HGVS_C:
                    # strip of leading 'n.' as redundant information
                    hgvs_c[i, j] = vals[ANNFidx.HGVS_C][2:]
                if hgvs_p is not None and n_vals > ANNFidx.HGVS_P:
                    # strip of leading 'p.' as redundant information
                    hgvs_p[i, j] = vals[ANNFidx.HGVS_P][2:]
                if cdna_pos is not None or cdna_length is not None and n_vals > ANNFidx.CDNA:
                    v = vals[ANNFidx.CDNA]
                    if v:
                        vv = v.split('/')
                        if cdna_pos is not None:
                            cdna_pos[i, j] = int(vv[0])
                        if cdna_length is not None and len(vv) > 1:
                            cdna_length[i, j] = int(vv[1])
                if cds_pos is not None or cds_length is not None and n_vals > ANNFidx.CDS:
                    v = vals[ANNFidx.CDS]
                    if v:
                        vv = v.split('/')
                        if cds_pos is not None:
                            cds_pos[i, j] = int(vv[0])
                        if cds_length is not None and len(vv) > 1:
                            cds_length[i, j] = int(vv[1])
                if aa_pos is not None or aa_length is not None and n_vals > ANNFidx.AA:
                    v = vals[ANNFidx.AA]
                    if v:
                        vv = v.split('/')
                        if aa_pos is not None:
                            aa_pos[i, j] = int(vv[0])
                        if aa_length is not None and len(vv) > 1:
                            aa_length[i, j] = int(vv[1])
                if distance is not None and n_vals > ANNFidx.DISTANCE:
                    v = vals[ANNFidx.DISTANCE]
                    if v:
                        distance[i, j] = int(v)

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
        if cdna_pos is not None:
            ann_chunk[ANN_CDNA_POS_FIELD] = cdna_pos
        if cdna_length is not None:
            ann_chunk[ANN_CDNA_LENGTH_FIELD] = cdna_length
        if cds_pos is not None:
            ann_chunk[ANN_CDS_POS_FIELD] = cds_pos
        if cds_length is not None:
            ann_chunk[ANN_CDS_LENGTH_FIELD] = cds_length
        if aa_pos is not None:
            ann_chunk[ANN_AA_POS_FIELD] = aa_pos
        if aa_length is not None:
            ann_chunk[ANN_AA_LENGTH_FIELD] = aa_length
        if distance is not None:
            ann_chunk[ANN_DISTANCE_FIELD] = distance

        if number == 1:
            for k in list(ann_chunk.keys()):
                ann_chunk[k] = ann_chunk[k].squeeze(axis=1)

        chunk.update(ann_chunk)
