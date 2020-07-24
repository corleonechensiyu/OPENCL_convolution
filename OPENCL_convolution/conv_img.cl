__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void conv2d(__read_only image2d_t  inputImage,
                    __write_only image2d_t  outputImage,
                    int Ndim, int Mdim,int filterWidth,__constant float *filter)
{
    int column = get_global_id(0);
    int row = get_global_id(1);  

    if(column >= Mdim || row >= Mdim)
        return;

    int2 coords;
    //int halfWidth = (int)(filterWidth / 2);
    float4 sum = {0, 0, 0, 0}; 
    int filterIdx = 0;    

    for (int i = 0; i < filterWidth; i++)
    {
        coords.y = row + i; 
        for (int j = 0; j < filterWidth; j++) 
        {
            coords.x = column + j; 
            float4 pixel;  
            //Use the coordinate (x, y) to do an element lookup in the 1D or 2D image object specified by image.
            //ʹ������(x, y)��ͼ��ָ����1D��2Dͼ�������ִ��Ԫ�ز���
            pixel = read_imagef(inputImage, SAMPLER, coords);
            sum.x += pixel.x * filter[filterIdx++];

        }
    }

    coords.x = column;
    coords.y = row;
    //����ɫֵд����ͼ��ָ����1D��2Dͼ�������������(x, y)ָ����λ�á�
    write_imagef(outputImage, coords, sum);

}

