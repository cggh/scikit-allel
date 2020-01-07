import os
import struct
import gzip
import zlib
import re
import numpy as np

# max value of int 32
MAX_INT32 = 0xffffffff >> 1

# max value of uint64
MAX_UINT64 = 0xffffffffffffffff


class VCFReader:
    def __init__(self, filename, index, region):
        self._chrom, self._begin, self._end = self.parse_region(region)
        self._chunk_begin = 0
        self._chunk_end = MAX_UINT64
        self._meta = b'#'
        self._header = True  # track whether we should read VCF header
        self._block = None
        self._parse_index(index)
        self._fileobj = open(filename, 'rb')
        self._load_block(0)  # load header

    @staticmethod
    def parse_region(region):
        """
        parse a region to chromosome, begin, end

        Parameters
        ----------
        region: string
            tabix style region string

        Returns
        -------
        chromosome: bytes
            contig name
        begin: int
            position begin
        end: int
            position end

        """
        try:
            chrom, begin, end = re.findall(r'^(\w+):*(\d*)-*(\d*)$', region).pop(0)
        except IndexError:
            raise ValueError('bad region string: {}'.format(region))
        begin = int(begin) - 1 if begin else 0
        end = int(end) if end else MAX_INT32
        if begin > 0 and end == MAX_INT32:
            end = begin + 1
        return chrom.encode('ascii'), begin, end

    @staticmethod
    def reg2bins(begin, end, n_lvls=5, min_shift=14):
        """
        generate key of bins which may overlap the given region,
        check out https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3042176/
        and https://samtools.github.io/hts-specs/tabix.pdf
        for more information.

        Parameters
        ----------
        begin: int
            chromosome position begin
        end: int
            chromosome position end
        n_lvls: int, optional
            cluster level, for tabix, set to 5
        min_shift: int, optional
            minimum shift, for tabix, set to 14

        Returns
        -------
        generator

        """
        begin, end = begin, end
        t, s = 0, min_shift + (n_lvls << 1) + n_lvls
        for l in range(n_lvls + 1):
            b, e = t + (begin >> s), t + (end >> s)
            n = e - b + 1
            for k in range(b, e + 1):
                yield k
                n += 1
            t += 1 << ((l << 1) + l)
            s -= 3

    def _parse_index(self, index):
        """
        read and parse the index file, set the virtual offset for the target region,
        check out https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3042176/
        and https://samtools.github.io/hts-specs/tabix.pdf
        for more information.

        Parameters
        ----------
        index: str
            path for the index file

        """
        with gzip.open(index, 'rb') as f:
            magic = f.read(4)
            if magic != b'TBI\x01':  # check magic
                raise RuntimeError('invalid tabix index file: {}'.format(index))
            header = np.frombuffer(f.read(4 * 8), dtype=np.int32)
            if header[1] != 2:  # check format
                raise RuntimeError('invalid VCF index file: {}'.format(index))
            # this is the comment marker use to determine VCF header, usually set to b'#'
            self._meta = chr(header[5]).encode('ascii')
            try:
                rid = f.read(header[7]).split(b'\x00').index(self._chrom)
            except ValueError:
                # contig name is not exist, read header only
                self._chunk_begin = -1
                return
            # seek to our target index
            for _ in range(rid):
                for _ in range(struct.unpack('<i', f.read(4))[0]):  # n_bins
                    f.seek(4, os.SEEK_CUR)  # bin_key
                    n_chunk = struct.unpack('<i', f.read(4))[0]  # n_chunk
                    f.seek(8 * 2 * n_chunk, os.SEEK_CUR)  # cnk_beg, cnk_end
                n_intv = struct.unpack('<i', f.read(4))[0]  # n_intv
                f.seek(8 * n_intv, os.SEEK_CUR)  # ioff
            # read our target index
            bidx = {bin_key: None for bin_key in self.reg2bins(self._begin, self._end)}
            for _ in range(struct.unpack('<i', f.read(4))[0]):  # n_bins
                bin_key = struct.unpack('<I', f.read(4))[0]  # bin
                n_chunk = struct.unpack('<i', f.read(4))[0]  # n_chunk
                if bin_key in bidx:
                    chunks = np.frombuffer(f.read(8 * 2 * n_chunk), dtype=np.uint64)
                    bidx[bin_key] = chunks.reshape(n_chunk, -1)  # cnk_beg, cnk_end
                else:
                    f.seek(8 * 2 * n_chunk, os.SEEK_CUR)  # cnk_beg, cnk_end
            n_intv = struct.unpack('<i', f.read(4))[0]  # n_intv
            ioff = np.frombuffer(f.read(8 * n_intv), dtype=np.uint64)  # ioff
        # linear index: record fall into 16kb(=16 * 2 ** 10) interval
        min_ioff = ioff[self._begin >> 14]
        # binning index: record cluster in large interval
        overlap = np.concatenate([chunks for bin_key, chunks in bidx.items() if chunks is not None])
        # coupled binning and linear indices, filter out low level bins
        chunk_begin, *_, chunk_end = np.sort(np.ravel(overlap[overlap[:, 0] >= min_ioff]))
        # convert to native int
        self._chunk_begin, self._chunk_end = chunk_begin.item(), chunk_end.item()

    def _load_block(self, offset):
        """
        load a BGZF block into buffer

        Parameters
        ----------
        offset: uint64
            the offset of the BGZF block, the higher 48 bits keep the real file offset
            of the start of the gzip block the byte falls in, the lower 16 bits store
            the offset of the byte inside the gzip block.
            check out https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3042176/
            and http://samtools.sourceforge.net/SAM1.pdf
            for more information.

        """
        if offset < 0:  # contig name is not exist
            return
        # the higher 48 bits keep the real file offset of the gzip block that byte falls in
        block_offset = offset >> 16
        if block_offset > (self._chunk_end >> 16):  # out of range, stop reading
            return
        self._fileobj.seek(block_offset)
        # read a BGZF block
        magic = self._fileobj.read(4)
        if not magic:  # end of file
            return
        if magic != b"\x1f\x8b\x08\x04":  # check whether this is a BGZF block
            raise RuntimeError("invalid BGZF block offset: {}!".format(block_offset))
        self._fileobj.seek(4 + 1 + 1, os.SEEK_CUR)  # seek useless MTIME, XFL, OS
        x_len = extra_len = struct.unpack("<H", self._fileobj.read(2))[0]
        bsize = None
        while extra_len > 0:
            subfield_identifier = self._fileobj.read(2)
            subfield_length = struct.unpack("<H", self._fileobj.read(2))[0]
            extra_len -= 4 + subfield_length
            # we need the b"BC" subfield only
            if subfield_identifier == b"BC" and subfield_length == 2:
                bsize = struct.unpack("<H", self._fileobj.read(2))[0]
                break
            else:
                self._fileobj.seek(subfield_length)
        if bsize is None:
            raise RuntimeError(
                "Can not found BSIZE subfield in BGZF block from offset {}!".format(block_offset)
            )
        elif extra_len > 0:  # skip left extra length
            self._fileobj.seek(self._fileobj.tell() + extra_len)
        # read and decompress the block data
        cdata = self._fileobj.read(bsize - x_len - 19)
        data = zlib.decompress(cdata, wbits=-15)
        # check out https://docs.python.org/3/library/zlib.html#zlib.crc32
        crc = zlib.crc32(data) & 0xffffffff
        if crc != struct.unpack("<I", self._fileobj.read(4))[0]:
            raise RuntimeError("invalid BGZF block offset: {}!".format(block_offset))
        # it is impossible that crc32 is correct but isize is not correct
        # isize = struct.unpack("<I", self.fileobj.read(4))[0]
        # assert len(data) == isize
        self._fileobj.seek(4, os.SEEK_CUR)
        # the lower 16 bits store the offset of the byte inside the gzip block
        block_begin = offset & 0xffff
        # shall we read to the end of this block?
        end_offset = self._chunk_end >> 16
        block_end = self._chunk_end & 0xffff if self._fileobj.tell() > end_offset else None
        self._block = data[block_begin:block_end]

    def close(self):
        self._fileobj.close()

    def readinto(self, b):
        """
        imitate the readinto function in BufferedStream class.
        This is th function used by FileInputStream.

        Parameters
        ----------
        b: bytes or bytearray
            pre-allocated, writeable bytes-like object.

        Returns
        -------
        num : int
            the number of bytes read

        """
        num = 0
        # read header
        while self._header and self._block and num < len(b):
            line_end = self._block.find(b'\n')  # no matter '\n' or '\r\n', this is the end
            read_len = min(len(b) - num, len(self._block) if line_end < 0 else line_end + 1)
            b[num:num + read_len] = self._block[:read_len]
            num += read_len
            self._block = self._block[read_len:]
            if not self._block:  # more to read, do not reach the end of header yet
                self._load_block(self._fileobj.tell() << 16)
            if not self._block.startswith(self._meta):  # end of header, load target region block
                self._header = False
                self._load_block(self._chunk_begin)
                break
        # read record
        while self._block and num < len(b):
            read_len = min(len(b) - num, len(self._block))
            b[num:num+read_len] = self._block[:read_len]
            num += read_len
            self._block = self._block[read_len:]
            if not self._block:
                self._load_block(self._fileobj.tell() << 16)
        return num
