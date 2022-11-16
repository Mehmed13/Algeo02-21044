#include <stdio.h>

int one = 1;
int overflow(void);

int main()
{
    int val = overflow();
    val += one;
    if (val != 15213)
    {
        printf("aaa");
    }
    else
    {
        printf("XXX");
    }
    return 0;
}

int overflow()
{
    char buf[4];
    int val, i = 0;
    while (scanf("%x", &val) != -1)
        buf[i++] = (char)val;
    return 15213;
}