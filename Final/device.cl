__kernel void DownSample (  __read_only image2d_t inputImg,
                            __write_only image2d_t outputImg,
							int SIZEX, int SIZEY,
							sampler_t sampler)
{
 
    int i = get_global_id (0);//index of the output image
    int j = get_global_id (1);
	int row = get_global_size(0)*SIZEX;//size of the input image
	int col = get_global_size(1)*SIZEY;

   // jump to the starting indexes
   int is = i * SIZEX;
   int js = j * SIZEY;
   //printf("(%d,%d)\n",i,j);

  
   float4 total = (float4) (0);
  
   for ( int x = 0; x < SIZEX; x++ ) {
        for ( int y = 0; y < SIZEY; y++ ) {
                total += read_imagef ( inputImg, sampler, (int2) ( is + x, js + y ) );
        }
    }

    total = (float4) ( total / (SIZEX*SIZEY) );
	
    write_imagef ( outputImg, (int2) ( i, j ), total );
}

__kernel void GaussianFilter(int filterWidth, float sigma,__global float * gaussBlurFilter,__global float * filtSum)
{
	float gauss[10];
	for(int i=0;i<filterWidth;i++)
	{
		gauss[i]=-1+i*(float)2/(filterWidth-1);
		
	}
	int filtIdx = get_global_id(0);
	int x=filtIdx%filterWidth;
	int y=filtIdx/filterWidth;
	gaussBlurFilter[filtIdx] = (float)1/(2*3.14159*sigma)*exp((-gauss[x]*gauss[x]-gauss[y]*gauss[y])/(2*sigma*sigma));
	//filtSum[filtIdx] = work_group_scan_inclusive_add(gaussBlurFilter[filtIdx]);
	//gaussBlurFilter[filtIdx]=(float)gaussBlurFilter[filtIdx]/filtSum[24];
	//printf(" After %f\n",filtSum[24]);
}

__kernel void GaussianBlur(__read_only image2d_t inputImg, __write_only image2d_t outputImg, sampler_t sampler, int filterWidth,__global float *gaussBlurFilter)
{

	// use global IDs for output coords
	int x = get_global_id(0); // columns
	int y = get_global_id(1); // rows
	int halfWidth = (int)(filterWidth/2); // auto-round nearest int ???
	float4 sum = (float4)(0);
	int filtIdx = 0; // filter kernel passed in as linearized buffer array
	int2 coords;
	for(int i = -halfWidth; i <= halfWidth; i++) // iterate filter rows
	{
		coords.y = y + i;
		for(int j = -halfWidth; j <= halfWidth; j++) // iterate filter cols
	  {
	  coords.x = x + j;
	  //float4 pixel = convert_float4(read_imageui(inputImg, sampler, coords)); // operate element-wise on all 3 color components (r,g,b)
	  float4 pixel = read_imagef(inputImg, sampler, coords); // operate element-wise on all 3 color components (r,g,b)
	  filtIdx++;
	  sum += pixel * (float4)(gaussBlurFilter[filtIdx],gaussBlurFilter[filtIdx],gaussBlurFilter[filtIdx],1.0f); // leave a-channel unchanged
	  }
     }
	//write resultant filtered pixel to output image
	coords = (int2)(x,y);
	//write_imageui(outputImg, coords, convert_uint4(sum));
	write_imagef(outputImg, coords, sum);
}

__kernel void DoG(__read_only image2d_t inputImg1,
				  __read_only image2d_t inputImg2,
				  __write_only image2d_t outputImg)
{ 


}

__kernel void Extrema(__read_only image2d_t inputImg1,
					 __read_only image2d_t inputImg2,
					 __read_only image2d_t inputImg3,
					 __global float2 * extrema)
{ 

}