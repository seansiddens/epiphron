#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <chrono>

using namespace std::chrono;

const int size = 1024 * 1024 * 32;

int main(int argc, char* argv[]) {
	// Initialize 
	auto instance = easyvk::Instance(true);
	auto device = instance.devices().at(0);
	std::cout << "Using device: " << device.properties.deviceName << "\n";
	auto maxWrkGrpCount = device.properties.limits.maxComputeWorkGroupCount;
	std::printf(
		"maxComputeWorkGroupCount: (%d, %d, %d)\n", 
		maxWrkGrpCount[0], 
		maxWrkGrpCount[1],
		maxWrkGrpCount[2]
	);

	// Create GPU buffers.
	std::cout << "Vector size: " << size << "\n";
	auto a = easyvk::Buffer(device, size);
	auto b = easyvk::Buffer(device, size);
	auto c = easyvk::Buffer(device, size);

	// Write initial values to the buffers.
	for (int i = 0; i < size; i++) {
		a.store(i, i);
		b.store(i, i + 1);
		c.store(i, 0);
	}
	std::vector<easyvk::Buffer> bufs = {a, b, c};

	std::vector<uint32_t> spvCode =
	#include "vect-add.cinit"
	;	
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(size);
	program.setWorkgroupSize(1);

	// Run the kernel.
	program.prepare("litmus_test");

	auto start_time = std::chrono::high_resolution_clock::now();
	auto timeLimit = 1.0;
    int numInvocations = 0;
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() < timeLimit * 1000.0) {
		// TODO: Is it fine to just repeatedly call it like this? 
		// Does the state of the command queue stay preserved? 
		program.run();
        numInvocations++;
    }

	// Validate the output.
	for (int i = 0; i < size; i++) {
		// std::cout << "c[" << i << "]: " << c.load(i) << "\n";
		assert(c.load(i) == a.load(i) + b.load(i));
	}

	std::cout << numInvocations << " kernel launches in " << timeLimit << " second(s).\n";


	// Cleanup.
	program.teardown();
	a.teardown();
	b.teardown();
	c.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
