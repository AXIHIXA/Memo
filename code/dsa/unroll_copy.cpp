/// Simplication of Tom Duff's tricky malloc copy code.
/// Handles the remainder in the first loop of an unrolled routine. 
void unrollCopy(unsigned char * dst, unsigned char * src, int len)
{
    // Differs from len / 8 + 1 when len is multiple of 8. 
    int n = (len + 7) / 8;

    switch (len % 8)
    {
        case 0: do { *dst++ = *src++;
        case 7:      *dst++ = *src++;
        case 6:      *dst++ = *src++;
        case 5:      *dst++ = *src++;
        case 4:      *dst++ = *src++;
        case 3:      *dst++ = *src++;
        case 2:      *dst++ = *src++;
        case 1:      *dst++ = *src++; } while (0 < --n);
    }
}
