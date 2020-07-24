__kernel void matrix_mult(int Ndim,int Mdim ,int filterWidth,
    __global const float* input,
    __global  float* output,
    __constant float *filter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float tmp;

    if ((x < Mdim) && (y < Mdim)) {
        tmp = 0.0;
        for (int r = 0; r < filterWidth; r++)
        {
            const int idxIntpm=(y+r)*Ndim+x;
            for (int c = 0; c < filterWidth; c++)
            {
               tmp+=filter[(r*filterWidth)+c]*input[idxIntpm+c];
               //tmp += dot(filter[(r*filterWidth)+c],input[idxIntpm+c]);
            }
        }
        output[y*Mdim+x]=tmp;
    }

}