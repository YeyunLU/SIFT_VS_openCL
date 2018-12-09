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

__kernel void GaussianFilter(int filterWidth, __global float* sigma,__global float * gaussBlurFilter,__global float * filtSum)
{
	float gauss[10];
	for(int i=0;i<filterWidth;i++)
	{
		gauss[i]=-1+i*(float)2/(filterWidth-1);
		
	}
	int filtIdx = get_global_id(0);
	int x=filtIdx%filterWidth;
	int y=filtIdx/filterWidth;
	gaussBlurFilter[filtIdx] = (float)1/(2*3.14159*sigma[0])*exp((-gauss[x]*gauss[x]-gauss[y]*gauss[y])/(2*sigma[0]*sigma[0]));
	filtSum[filtIdx] = work_group_scan_inclusive_add(gaussBlurFilter[filtIdx]);
	gaussBlurFilter[filtIdx]=(float)gaussBlurFilter[filtIdx]/filtSum[24];
	//printf(" After %f\n",filtSum[24]);
}

__kernel void GaussianBlur(__read_only image2d_t inputImg, __write_only image2d_t outputImg, sampler_t sampler, int filterWidth,__global float *gaussBlurFilter)
{

	// use global IDs for output coords
	int x = get_global_id(0); // columns
	int y = get_global_id(1); // rows
	//int z = get_global_id(2); // sigmas

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
				  __write_only image2d_t outputImg,
				  sampler_t sampler)
{ 
	

}

__kernel void Extrema(__read_only image2d_t preImg,
					  __read_only image2d_t curImg,
					  __read_only image2d_t nextImg,
					  __global uint2 * extrema,
					  sampler_t sampler)
{ 
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	float4 point = read_imagef(preImg, sampler,(int2)(x,y));
	float4 p1 = read_imagef(preImg, sampler, (int2)(x,y-1)); //compare with 26 points
	float4 p2 = read_imagef(preImg, sampler, (int2)(x,y+1));
	float4 p3 = read_imagef(preImg, sampler, (int2)(x-1,y-1));
	float4 p4 = read_imagef(preImg, sampler, (int2)(x-1,y));
	float4 p5 = read_imagef(preImg, sampler, (int2)(x-1,y+1));
	float4 p6 = read_imagef(preImg, sampler, (int2)(x+1,y-1));
	float4 p7 = read_imagef(preImg, sampler, (int2)(x+1,y));
	float4 p8 = read_imagef(preImg, sampler, (int2)(x+1,y+1));
	float4 c1 = read_imagef(curImg, sampler, (int2)(x,y-1));
	float4 c2 = read_imagef(curImg, sampler, (int2)(x,y));
	float4 c3 = read_imagef(curImg, sampler, (int2)(x,y+1));
	float4 c4 = read_imagef(curImg, sampler, (int2)(x-1,y-1));
	float4 c5 = read_imagef(curImg, sampler, (int2)(x-1,y));
	float4 c6 = read_imagef(curImg, sampler, (int2)(x-1,y+1));
	float4 c7 = read_imagef(curImg, sampler, (int2)(x+1,y-1));
	float4 c8 = read_imagef(curImg, sampler, (int2)(x+1,y));
	float4 c9 = read_imagef(curImg, sampler, (int2)(x+1,y+1));
	float4 n1 = read_imagef(nextImg, sampler, (int2)(x,y-1));
	float4 n2 = read_imagef(nextImg, sampler, (int2)(x,y));
	float4 n3 = read_imagef(nextImg, sampler, (int2)(x,y+1));
	float4 n4 = read_imagef(nextImg, sampler, (int2)(x-1,y-1));
	float4 n5 = read_imagef(nextImg, sampler, (int2)(x-1,y));
	float4 n6 = read_imagef(nextImg, sampler, (int2)(x-1,y+1));
	float4 n7 = read_imagef(nextImg, sampler, (int2)(x+1,y-1));
	float4 n8 = read_imagef(nextImg, sampler, (int2)(x+1,y));
	float4 n9 = read_imagef(nextImg, sampler, (int2)(x+1,y+1));
	//if(point>p1 && point>p2 && point>p3 && point>p4 && point>p5 && point>p6 && point>p7 && point>p8
	//&&point>c1 && point>c2 && point>c3 && point>c4 && point>c5 && point>c6 && point>c7 && point>c8 && point>c9
	//&&point>n1 && point>n2 && point>n3 && point>n4 && point>n5 && point>n6 && point>n7 && point>n8 && point>n9)
	//extrema[0]=(uint2)(x,y);

	
}
