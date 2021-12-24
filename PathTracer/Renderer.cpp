#include "Renderer.h"
#include <optix_function_table_definition.h>

namespace pt {

	extern "C" char embedded_ptx_code[];

	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		void* data;
	};

	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		MissData data;
	};

	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		TriangleMeshSBTData data;
	};

	void pt::Renderer::update_subframe_index()
	{
		++launchParams.subframe_index;
	}

	pt::Renderer::Renderer(const Model* model)
		: model(model)
	{	
		initOptix();
		createContext();
		createModule();
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		launchParams.traversable = buildAccel();
		createPipeline();
		buildSBT();
		setLight();
		launchParamsBuffer.alloc(sizeof(launchParams));
		launchParams.spp = 10;
		launchParams.light_samples = 1;
		launchParams.P_RR = 0.8f;
		launchParams.subframe_index = 0;
	}

	void pt::Renderer::render()
	{
		if (launchParams.frame.size.x == 0)
			return;
		launchParamsBuffer.upload(&launchParams, 1);

		OPTIX_CHECK(optixLaunch(
			pipeline,
			stream,
			launchParamsBuffer.d_pointer(),
			launchParamsBuffer.sizeBytes,
			&sbt,
			launchParams.frame.size.x,
			launchParams.frame.size.y,
			1
		));
		OptixDenoiserParams denoiserParams;
		denoiserParams.denoiseAlpha = 1;
		denoiserParams.hdrIntensity = (CUdeviceptr)0;
		denoiserParams.blendFactor = 0.0f;

		// -------------------------------------------------------
		OptixImage2D inputLayer;
		inputLayer.data = colorBuffer.d_pointer();
		/// Width of the image (in pixels)
		inputLayer.width = launchParams.frame.size.x;
		/// Height of the image (in pixels)
		inputLayer.height = launchParams.frame.size.y;
		/// Stride between subsequent rows of the image (in bytes).
		inputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
		/// Stride between subsequent pixels of the image (in bytes).
		/// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
		inputLayer.pixelStrideInBytes = sizeof(float4);
		/// Pixel format.
		inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		// -------------------------------------------------------
		OptixImage2D outputLayer;
		outputLayer.data = denoisedBuffer.d_pointer();
		/// Width of the image (in pixels)
		outputLayer.width = launchParams.frame.size.x;
		/// Height of the image (in pixels)
		outputLayer.height = launchParams.frame.size.y;
		/// Stride between subsequent rows of the image (in bytes).
		outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
		/// Stride between subsequent pixels of the image (in bytes).
		/// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
		outputLayer.pixelStrideInBytes = sizeof(float4);
		/// Pixel format.
		outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		OptixDenoiserGuideLayer denoiserGuideLayer = {};

		OptixDenoiserLayer denoiserLayer = {};
		denoiserLayer.input = inputLayer;
		denoiserLayer.output = outputLayer;

		OPTIX_CHECK(optixDenoiserInvoke(denoiser,
			/*stream*/0,
			&denoiserParams,
			denoiserState.d_pointer(),
			denoiserState.sizeBytes,
			&denoiserGuideLayer,
			&denoiserLayer, 1,
			/*inputOffsetX*/0,
			/*inputOffsetY*/0,
			denoiserScratch.d_pointer(),
			denoiserScratch.sizeBytes));
		CUDA_SYNC_CHECK();
	}

	void pt::Renderer::resize(const vec2i& newSize)
	{
		if (denoiser) {
			OPTIX_CHECK(optixDenoiserDestroy(denoiser));
		}

		OptixDenoiserOptions denoiserOptions = {};

		OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
		OptixDenoiserSizes denoiserReturnSize;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y, &denoiserReturnSize));
		denoiserScratch.resize(std::max(denoiserReturnSize.withOverlapScratchSizeInBytes,
			denoiserReturnSize.withoutOverlapScratchSizeInBytes));

		denoiserState.resize(denoiserReturnSize.stateSizeInBytes);

		denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));

		if (newSize.x == 0 || newSize.y == 0)
			return;
		colorBuffer.resize(newSize.x * newSize.y * sizeof(float4));
		accum_color_buffer.resize(newSize.x * newSize.y * sizeof(float4));
		launchParams.frame.size = newSize;
		launchParams.frame.colorBuffer = (float4*)colorBuffer.d_pointer();
		launchParams.frame.accum_color = (float4*)accum_color_buffer.d_pointer();
		setCamera(lastSetCamera);

		OPTIX_CHECK(
			optixDenoiserSetup(
				denoiser, 0,
				newSize.x, newSize.y,
				denoiserState.d_pointer(),
				denoiserState.sizeBytes,
				denoiserScratch.d_pointer(),
				denoiserScratch.sizeBytes
			)
		);
	}

	void pt::Renderer::downloadPixels(vec4f h_pixels[])
	{
		denoisedBuffer.download(h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);
	}

	void pt::Renderer::setCamera(const Camera& camera)
	{
		lastSetCamera = camera;
		launchParams.camera.dir = normalize(camera.at - camera.pos);
		launchParams.camera.pos = camera.pos;
		const float aspect_ratio = launchParams.frame.size.x / (float)launchParams.frame.size.y;
		launchParams.camera.horizontal= normalize(cross(launchParams.camera.dir, camera.up)) * aspect_ratio;
		launchParams.camera.vertical = normalize(cross(launchParams.camera.horizontal, launchParams.camera.dir));
	}

	void Renderer::setLight()
	{
		launchParams.light.corner = vec3f(-0.24f, 1.98f, -0.22f);
		launchParams.light.emission = vec3f(47.0f, 38.0f, 31.0f);
		launchParams.light.v1 = vec3f(0.f, 0.f, 0.38);
		launchParams.light.v2 = vec3f(0.47f, 0.f, 0.f);
		launchParams.light.normal = -normalize(cross(launchParams.light.v1, launchParams.light.v2));
	}

	void Renderer::initOptix()
	{
		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0) {
			throw std::runtime_error("no CUDA capable devices found");
		}

		OPTIX_CHECK(optixInit());
	}

	static void context_log_cb(
		unsigned int level,
		const char* tag,
		const char* msg,
		void*)
	{
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, msg);
	}

	void Renderer::createContext()
	{
		const int deviceID = 0;
		CUDA_CHECK(SetDevice(deviceID));
		CUDA_CHECK(StreamCreate(&stream));

		cudaGetDeviceProperties(&deviceProps, deviceID);
		std::cout << "running on device " << deviceProps.name << std::endl;

		CUresult cuRes = cuCtxGetCurrent(&cudaContext);
		if (cuRes != CUDA_SUCCESS) {
			fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
		}

		OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
	}

	void Renderer::createModule()
	{
		moduleCompileOptions.maxRegisterCount = 50;
		moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

		pipelineCompileOptions = {};
		pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineCompileOptions.usesMotionBlur = false;
		pipelineCompileOptions.numPayloadValues = 2;
		pipelineCompileOptions.numAttributeValues = 2;
		pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		pipelineLinkOptions.maxTraceDepth = 2;

		const std::string ptxCode = embedded_ptx_code;

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			ptxCode.c_str(),
			ptxCode.size(),
			log, &sizeof_log,
			&module
		));
	}

	void Renderer::createRaygenPrograms()
	{
		raygenPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgDesc.raygen.module = module;
		pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&raygenPGs[0]
		));
	}

	void Renderer::createMissPrograms()
	{
		missPGs.resize(2);
		{
			OptixProgramGroupOptions pgOptions = {};
			OptixProgramGroupDesc pgDesc = {};
			pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			pgDesc.miss.module = module;
			pgDesc.miss.entryFunctionName = "__miss__radiance";

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(
				optixContext,
				&pgDesc,
				1,
				&pgOptions,
				log,
				&sizeof_log,
				&missPGs[0]
			));
		}
		{
			OptixProgramGroupOptions pgOptions = {};
			OptixProgramGroupDesc pgDesc = {};
			pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			pgDesc.miss.module = module;
			pgDesc.miss.entryFunctionName = "__miss__occlusion";

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(
				optixContext,
				&pgDesc,
				1,
				&pgOptions,
				log,
				&sizeof_log,
				&missPGs[1]
			));
		}
	}

	void Renderer::createHitgroupPrograms()
	{
		hitgroupPGs.resize(2);
		{
			OptixProgramGroupOptions pgOptions = {};
			OptixProgramGroupDesc pgDesc = {};
			pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			pgDesc.hitgroup.moduleCH = module;
			pgDesc.hitgroup.moduleAH = module;
			pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
			pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(
				optixContext,
				&pgDesc,
				1,
				&pgOptions,
				log,
				&sizeof_log,
				&hitgroupPGs[0]
			));
		}
		{
			OptixProgramGroupOptions pgOptions = {};
			OptixProgramGroupDesc pgDesc = {};
			pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			pgDesc.hitgroup.moduleCH = module;
			pgDesc.hitgroup.moduleAH = module;
			pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
			pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(
				optixContext,
				&pgDesc,
				1,
				&pgOptions,
				log,
				&sizeof_log,
				&hitgroupPGs[1]
			));
		}
	}

	void Renderer::createPipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		for (auto pg : raygenPGs) {
			programGroups.push_back(pg);
		}
		for (auto pg : missPGs) {
			programGroups.push_back(pg);
		}
		for (auto pg : hitgroupPGs) {
			programGroups.push_back(pg);
		}

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			optixContext,
			&pipelineCompileOptions,
			&pipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log,
			&sizeof_log,
			&pipeline
		));

		OPTIX_CHECK(optixPipelineSetStackSize(
			pipeline,
			2 * 1024,
			2 * 1024,
			2 * 1024,
			1
		));
	}

	void Renderer::buildSBT()
	{
		// raygen records
		std::vector<RaygenRecord> raygenRecords;
		for (int i = 0; i < raygenPGs.size(); i++) {
			RaygenRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
			rec.data = nullptr;
			raygenRecords.push_back(rec);
		}
		raygenRecordsBuffer.alloc_and_upload(raygenRecords);
		sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

		// miss records
		std::vector<MissRecord> missRecords;
		for (int i = 0; i < missPGs.size(); i++) {
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
			rec.data.bg_color = vec3f(0.f);
			missRecords.push_back(rec);
		}
		missRecordsBuffer.alloc_and_upload(missRecords);
		sbt.missRecordBase = missRecordsBuffer.d_pointer();
		sbt.missRecordStrideInBytes = sizeof(MissRecord);
		sbt.missRecordCount = (int)missRecords.size();

		// hitgroup records
		int numObjects = (int)model->meshes.size();
		std::vector<HitGroupRecord> hitGroupRecords;
		for (int meshID = 0; meshID < numObjects; meshID++) {
			HitGroupRecord rec;
			{
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
				rec.data.color = model->meshes[meshID]->diffuse;
				rec.data.emission = model->meshes[meshID]->emission;
				rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
				rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
				hitGroupRecords.push_back(rec);
			}
			{
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[1], &rec));
				hitGroupRecords.push_back(rec);
			}
		}
		hitgroupRecordsBuffer.alloc_and_upload(hitGroupRecords);
		sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
		sbt.hitgroupRecordCount = (int)hitGroupRecords.size();
	}

	OptixTraversableHandle Renderer::buildAccel()
	{
		vertexBuffer.resize(model->meshes.size());
		indexBuffer.resize(model->meshes.size());

		OptixTraversableHandle asHandle{ 0 };

		// triangle input
		std::vector<OptixBuildInput> triangleInput(model->meshes.size());
		std::vector<CUdeviceptr> d_vertices(model->meshes.size());
		std::vector<CUdeviceptr> d_indices(model->meshes.size());
		std::vector<uint32_t> triangleInputFlags(model->meshes.size());

		for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
			// upload the model to the device: the builder
			TriangleMesh& mesh = *model->meshes[meshID];
			vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
			indexBuffer[meshID].alloc_and_upload(mesh.index);

			triangleInput[meshID] = {};
			triangleInput[meshID].type
				= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			// create local variables, because we need a *pointer* to the
			// device pointers
			d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
			d_indices[meshID] = indexBuffer[meshID].d_pointer();

			triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
			triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
			triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

			triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
			triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
			triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

			triangleInputFlags[meshID] = 0;

			// in this example we have one SBT entry, and no per-primitive
			// materials:
			triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
			triangleInput[meshID].triangleArray.numSbtRecords = 1;
			triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
		}
		// ==================================================================
		// BLAS setup
		// ==================================================================

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
			| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
			;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage
		(optixContext,
			&accelOptions,
			triangleInput.data(),
			(int)model->meshes.size(),  // num_build_inputs
			&blasBufferSizes
		));

		// ==================================================================
		// prepare compaction
		// ==================================================================

		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();

		// ==================================================================
		// execute build (main stage)
		// ==================================================================

		CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(optixContext,
			/* stream */0,
			&accelOptions,
			triangleInput.data(),
			(int)model->meshes.size(),
			tempBuffer.d_pointer(),
			tempBuffer.sizeBytes,

			outputBuffer.d_pointer(),
			outputBuffer.sizeBytes,

			&asHandle,

			&emitDesc, 1
		));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);
		asBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext,
			/*stream:*/0,
			asHandle,
			asBuffer.d_pointer(),
			asBuffer.sizeBytes,
			&asHandle));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		outputBuffer.free(); // << the UNcompacted, temporary output buffer
		tempBuffer.free();
		compactedSizeBuffer.free();

		return asHandle;
	}

}
