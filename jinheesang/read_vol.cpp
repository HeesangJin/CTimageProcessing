#include <cstdio>
#include <cstdint>

int main(void)
{
    FILE    *fp_sour;
    FILE    *fp_dest;
    unsigned char buff[2048];
    size_t   n_size;
    char c;
    
    
    fp_sour  = fopen("./Pramook_black_velvet_3.03um_80kV_down.vol" , "rb");
    n_size = fread( buff, 1, 2048, fp_sour);
//    for(int i=0; i<n_size; i++){
//        printf("%d: %d\n", i+1 ,buff[i]);
//    }
    int i=0;
    printf("%d: %c\n", i+1 ,buff[i]);
    i++;
    printf("%d: %c\n", i+1 ,buff[i]);
    i++;
    printf("%d: %c\n", i+1 ,buff[i]);
    i++;
    printf("%d: %d\n", i+1 ,buff[i]);
    i++;
    
    int final;
    
    final=0;
    final |= ( (int)(buff[i++])  );
    final |= ( (int)(buff[i++]) << 8 );
    final |= ( (int)(buff[i++]) <<  16 );
    final |= ( (int)(buff[i++]) <<  24  );
    printf("%d: %d\n", i+1 ,final);
    
    
    final=0;
    final |= ( (int)(buff[i++])  );
    final |= ( (int)(buff[i++]) << 8 );
    final |= ( (int)(buff[i++]) <<  16 );
    final |= ( (int)(buff[i++]) <<  24  );
    printf("number of x: %d\n", final);
    
    
    final=0;
    final |= ( (int)(buff[i++])  );
    final |= ( (int)(buff[i++]) << 8 );
    final |= ( (int)(buff[i++]) <<  16 );
    final |= ( (int)(buff[i++]) <<  24  );
    printf("number of y: %d\n", final);
    
    final=0;
    final |= ( (int)(buff[i++])  );
    final |= ( (int)(buff[i++]) << 8 );
    final |= ( (int)(buff[i++]) <<  16 );
    final |= ( (int)(buff[i++]) <<  24  );
    printf("number of z: %d\n", final);
    
    final=0;
    final |= ( (int)(buff[i++])  );
    final |= ( (int)(buff[i++]) << 8 );
    final |= ( (int)(buff[i++]) <<  16 );
    final |= ( (int)(buff[i++]) <<  24  );
    printf("number of channels: %d\n", final);
    
    ////////////////////////////////////////////////////////
    
    int floatingNum=0;
    printf("[%d]\n", buff[i]);
    floatingNum |= ( buff[i++]  );
    printf("[%d]\n", buff[i]);
    floatingNum |= ( buff[i++] << 8 );
    printf("[%d]\n", buff[i]);
    floatingNum |= ( buff[i++] <<  16 );
    printf("[%d]\n", buff[i]);
    floatingNum |= ( buff[i++] <<  24  );
    printf("xmin: %f\n", floatingNum);
    
    
    
    fclose(fp_sour);
    return 0;
}