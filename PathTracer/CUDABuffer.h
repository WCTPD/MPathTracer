#pragma once

#include "optix7.h"
#include <assert.h>
#include <vector>

namespace pt {

	class CUDABuffer
	{
	public:
		size_t sizeBytes{ 0 };
		void* d_ptr{ nullptr };

		inline CUdeviceptr d_pointer() const {
			return (CUdeviceptr)d_ptr;
		}

		void alloc(size_t size) {
			assert(d_ptr == nullptr);
			sizeBytes = size;
			CUDA_CHECK(Malloc((void**)&d_ptr, sizeBytes));
		}

		void free() {
			CUDA_CHECK(Free(d_ptr));
			d_ptr = nullptr;
			sizeBytes = 0;
		}

		template<typename T>
		void upload(const T* t, size_t count) {
			assert(d_ptr != nullptr);
			assert(sizeBytes == count * sizeof(T));
			CUDA_CHECK(Memcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
		}

		template<typename T>
		void alloc_and_upload(const std::vector<T>& vt) {
			alloc(vt.size() * sizeof(T));
			upload((const T*)vt.data(), vt.size());
		}

		template<typename T>
		void download(T* t, size_t count) {
			assert(d_ptr != nullptr);
			assert(sizeBytes == sizeof(T) * count);
			CUDA_CHECK(Memcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		void resize(size_t size) {
			if (d_ptr) {
				free();
			}
			alloc(size);
		}
	};

}
