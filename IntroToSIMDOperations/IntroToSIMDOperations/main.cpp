/*
Title: Intro to SIMD Operations
File Name: main.cpp
Copyright © 2016
Original authors: Luna Meier
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:
This example is meant to be a demonstration of how to use SIMD operations in 
Visual Studios.

References:
https://msdn.microsoft.com/en-us/library/t467de55(v=vs.90).aspx
*/

#include <intrin.h> //Need to include this for the SIMD ops
#include <cstdio>
#include <conio.h>
#include <string>

// SIMD stands for Single Instruction, Multiple Data.  It's also often known as vectorization.
// It's an attempt to speed up calculations by allowing for multiple sets of data to be operated
// on in parallel, so long as the operation to apply to all of them is the same.
//
// This tutorial is going to show you some intrinsics that allow for processors to perform these
// operations.  We're going to be using an operation set known as SSE.
//
// SSE stands for Streaming SIMD Extensions.  It's a set of instructions that allow for
// us to operate in parallel on multiple sets of data.  We'll be using the intrinsics that have been
// set up for us to utilize this power.

void printFloats(float* output, int length);

int main(){


	// Let's start by allocating some memory.
	// SSE SIMD operations require our memory be 16 bit aligned.
	// We can use _aligned_malloc for this purpose.

	float* foo = (float*)_aligned_malloc(sizeof(float) * 8, 16);
	float* bar = (float*)_aligned_malloc(sizeof(float) * 8, 16);
	// Let's initialize our values so we can pretend with some of the data.
	for (int i = 0; i < 8; ++i){
		foo[i] = i;
		bar[i] = i * 0.1f;
	}

	//Now let's use SIMD operations to multiply all of these together and store the results into a third buffer.

	float* results = (float*)_aligned_malloc(sizeof(float) * 8, 16);

	for (int i = 0; i < 8; i += 4){//remember to advance by fours since we're doing things four at a time.

		// Let's start with how to write these functions out.
		// All of these intrinsics start with _mm_
		// then comes the function name
		// then comes the data type, written as _ps()
		// There are other data types than _ps, but we won't look at those
		// since we just want to deal with floats.  (ps means single precision). 
		
		// The data type simd ops work with is the __m128 data type.  It's a 128 bit buffer
		// that stores in this case 4 floats.

		// We'll start with the load operation.  You pass in a pointer to a set of floats, and it
		// will load four of them into the buffer.
		__m128 a = _mm_load_ps(foo + i);

		// You CAN actually access the individual members of this buffer, as follows.
		printf("%f ",a.m128_f32[0]);
		// Most of the time you will not want to though.  

		// Let's load the other one now.

		__m128 b = _mm_load_ps(bar + i);

		// Let's add them together.
		__m128 c = _mm_add_ps(a, b);

		// And store them
		_mm_store_ps(results + i, c);
	}

	printf("\n\n");

	//And as you'll see, everything works out.
	printFloats(results, 8);

	// Now you should note that in a normal circumstance
	// you wouldn't want to write everything out like that for
	// a simple operation.

	// It's relatively easy to just write
	//_mm_store_ps(results + i, _mm_add_ps(_mm_load_ps(foo + i), _mm_load_ps(bar + i)));

	// Let's do something with SIMD.
	// Ooh, I know!  Let's do dot products.  To make a point.

	// So a lot of people try to use SIMD to do dot product as follows.

	float* vectors = (float*)_aligned_malloc(sizeof(float) * 8, 16);
	vectors[0] = 1.0f;//x  //observer the AoS format (Array of Structs)
	vectors[1] = 1.0f;//y
	vectors[2] = 1.0f;//z
	vectors[3] = 1.0f;//w

	vectors[4] = 2.0f;//x
	vectors[5] = 2.0f;//y
	vectors[6] = 2.0f;//z
	vectors[7] = 2.0f;//w

	__m128 a = _mm_load_ps(vectors);
	__m128 b = _mm_load_ps(vectors + 4);

	__m128 c = _mm_mul_ps(a, b); // okay now you have x^2, y^2, z^2, w^2
	//...now what?

	//How do you add them all together?  Guess we have to pull them out.
	_mm_store_ps(results, c);
	float product = (results[0] + results[1]) + (results[2] + results[3]);

	// Woohoo!  Dot product done, right?
	printf("%f \n\n", product);

	// Okay how many operations did that take?
	// The multiplication is 1, and if the processor is pipelining correctly probably 2 adds.
	// (your processor can do the two adds in parentheses at the same time, actually)

	// So 3 ops for 1 dot product.  Amazing.

	// Okay, so before we continue technically there's an SSE4.1 Instruction that will
	// just do the dot product of two vec4's for you, but even that is going to be less
	// effective than what I'm about to show you.

	// You're thinking about this dot product incorrectly.  Most of the time when do you
	// REALLY only need to do JUST one dot product?  It's actually not overly common.
	// Most of the time you want to dot product a whole set of numbers.  So let's construct
	// the data that way.

	float* x = (float*)_aligned_malloc(sizeof(float) * 64, 16); //64 floats, a much higher number.
	float* y = (float*)_aligned_malloc(sizeof(float) * 64, 16);
	float* z = (float*)_aligned_malloc(sizeof(float) * 64, 16);
	float* w = (float*)_aligned_malloc(sizeof(float) * 64, 16);//SoA, mmm, delicious cache coherence.

	float* dp = (float*)_aligned_malloc(sizeof(float) * 64, 16); //Need a results buffer big enough.

	// So we're going to do the dot product of the first vec4 (x[0], y[0], z[0], w[0])
	// against every other vector in this set.

	for (int i = 1; i < 64; ++i){
		x[i] = 0.1f;
		y[i] = 0.1f;
		z[i] = 0.1f;
		w[i] = 0.1f;
	}

	x[0] = 1.0f;
	y[0] = 1.0f;
	w[0] = 1.0f;
	z[0] = 1.0f;
	
	// First let's set up some buffers.

	__m128 x1 = _mm_load1_ps(x + 0); // Wait a second because if you look closely you'll
	// notice this is a different instruction than before.  load1 actually loads into the
	// buffer only one value, four times.  "But isn't SIMD about doing 4 things at once?"
	//
	// First off, SIMD is about using one instruction against ANY amount of data at once.
	// There are some cool new instructions I'm not going to cover here that allow you to
	// do 8 floats at once, for example.  Those are a little less supported as far as I can
	// tell.  Maybe I'll write a tutorial specifically for them (there's not too much different).
	//
	// Second of all, we're going to do 4 dot products against this one vector. 
	// So we actually need all of this buffer to be the same.
	// Plus we only need to load it once.

	__m128 y1 = _mm_load1_ps(y + 0);
	__m128 z1 = _mm_load1_ps(z + 0);
	__m128 w1 = _mm_load1_ps(w + 0);


	for (int i = 0; i < 64; i += 4){

		// Let's do the multiplication now.
		__m128 dx = _mm_mul_ps(x1, _mm_load_ps(x + i));
		__m128 dy = _mm_mul_ps(y1, _mm_load_ps(y + i));
		__m128 dz = _mm_mul_ps(z1, _mm_load_ps(z + i));
		__m128 dw = _mm_mul_ps(w1, _mm_load_ps(w + i));

		// Now let's do some addition.  We'll store all
		// the results into dx, so we're not making
		// any more variables.

		dx = _mm_add_ps(dx, dy);
		dz = _mm_add_ps(dz, dw);
		dx = _mm_add_ps(dx, dz);

		//That's four dot products.
		_mm_store_ps(dp + i, dx);

	}

	printFloats(dp, 8);
	// Cool.  Even got the squared Mag for free too out of it.
	
	// So let's count operations now.
	// Let's be really conservative with this one, and assume that
	// the processor is not pipelining any of these operations.
	// That's 4 multiplications, and 3 additions.
	// So 7 operations.
	//
	// But that was 7 operations for /4/ dot products.
	// That comes out to 7/4 operations per dot product.
	// Sure, this requires you to do at least four dot products to
	// be worth doing.  But like I said, most of the time you are doing
	// four at once.  Just layout the program correctly to take advantage of
	// the sheer power provided.

	// So hopefully this provided some insight on how to use the SIMD
	// intrinsics included.

	_getch();
}

// Just a simple printing function.
void printFloats(float* output, int length){
	for (int i = 0; i < length; ++i){
		std::printf("%f", output[i]);
		std::printf("\n");
	}

	std::printf("\n");
}