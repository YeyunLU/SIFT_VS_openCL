__constant float4 grayscale = { 0.2989f, 0.5870f, 0.1140f, 0 };

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

__kernel void GaussianFilter(int filterWidth, 
                             __global float* sigma,
							 __global float * gaussBlurFilter,
							 __global float * filtSum)
{
	float gauss[10];
	for(int i=0;i<filterWidth;i++)
	{
		gauss[i]=-1+i*(float)2/(filterWidth-1);
		
	}
	int filtIdx = get_local_id(0);
	int filtSize = get_local_size(0);
	int x = filtIdx%filterWidth;
	int y = filtIdx/filterWidth;
	int z = get_global_id(1);
	//printf("%d\n",z*filtSize+filtIdx);
	gaussBlurFilter[z*filtSize+filtIdx] = (float)1/(2*3.14159*sigma[z])*exp((-gauss[x]*gauss[x]-gauss[y]*gauss[y])/(2*sigma[z]*sigma[z]));
	filtSum[z*filtSize+filtIdx] = work_group_scan_inclusive_add(gaussBlurFilter[z*filtSize+filtIdx]);
	gaussBlurFilter[z*filtSize+filtIdx]=(float)gaussBlurFilter[z*filtSize+filtIdx]/filtSum[z*filtSize+24];
	//printf(" After %d, %f\n",z,filtSum[z*filtSize+24]);
}

__kernel void GaussianBlur(__read_only image2d_t inputImg, 
                           __write_only image2d_array_t outputImg,
						   sampler_t sampler, int filterWidth,
						   __global float *gaussBlurFilter)
{

	// use global IDs for output coords
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // sigmas

	int halfWidth = (int)(filterWidth/2); // auto-round nearest int ???
	int filtSize = filterWidth*filterWidth;
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
	  sum += pixel * (float4)(gaussBlurFilter[z*filtSize+filtIdx],gaussBlurFilter[z*filtSize+filtIdx],gaussBlurFilter[z*filtSize+filtIdx],1.0f); // leave a-channel unchanged
	  //printf("sum : %f,%f,%f,%f\n",sum.x,sum.y,sum.z,sum.w);
	  }
     }
	//write resultant filtered pixel to output image
	int4 coords2 = (int4)(x,y,z,0);
	//write_imageui(outputImg, coords, convert_uint4(sum));
	write_imagef(outputImg, coords2, sum);
}


__kernel void DifferenceOfGaussian(__read_only image2d_array_t inputImg, __write_only image2d_array_t outputImg, sampler_t sampler)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int4 coords_dog = (int4)(x,y,z,0);
	int4 coords = (int4)(x,y,z+1,0);

	float4 pixel1 = read_imagef(inputImg, sampler, coords_dog);
	float4 pixel2 = read_imagef(inputImg, sampler, coords);


	float4 DoG = pixel1 - pixel2;
	write_imagef(outputImg, coords_dog, DoG);

}

__kernel void Extrema(__read_only image2d_array_t DoGImg,
					  __global uint2 * extrema,
					  sampler_t sampler,
					  __global int *index)
{ 
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // index of image
	float4 rgba;
	int idx = 0;
	__local float p[27];
	if(x!=0&&x!=511&&y!=0&&y!=511)
	{
		for(int i=0;i<3;i++)
		{
			for(int j=-1;j<=1;j++)
			{ 
				for(int k=-1;k<=1;k++)
				{
					rgba = read_imagef(DoGImg,sampler,(int4)(x+j,y+k,z+i,0));
					//printf("%f\n",rgba.x);
					p[i*9+(j+1)*3+(k+1)] = (float) dot(grayscale, rgba);
				}			   
			}		
		}
		//printf("p[13]:%f\n",p[13]);
		if(p[13]>p[0] && p[13]>p[1] && p[13]>p[2] && p[13]>p[3] && p[13]>p[4] && p[13]>p[5]
		&& p[13]>p[6] && p[13]>p[7] && p[13]>p[8] && p[13]>p[9] && p[13]>p[10] && p[13]>p[11]
		&& p[13]>p[12] && p[13]>p[14] && p[13]>p[15] && p[13]>p[16] && p[13]>p[17] && p[13]>p[18]
		&& p[13]>p[19] && p[13]>p[20] && p[13]>p[21] && p[13]>p[22] && p[13]>p[23] && p[13]>p[24]
		&& p[13]>p[25] && p[13]>p[26])
		{
		
			idx = atomic_add(index,1);
			printf("(%d,%d,%d)\n",x,y,z);
		    extrema[idx]=(uint2)(x,y);
		}	   
	}
	
	
	
}


