# nano_rev
 Acceleration of Identity block of NanoReviser neural network



This project requires the libfaketime library to generate IP without issue and is built using Vitis 2019.2. It requires the utils.mk file, which is included in this repo. I believe the makefile requires other files present in the Xilinx Vitis_Accel_Examples repository, so the main folder in the nano_rev folder should be placed in that repository to ensure it can build and execute properly. The project can be built from the main folder with the command “faketime 'last year' make all TARGET=hw DEVICE=xilinx_u200_xdma_201830_2”. Afterwards it can be run with “faketime 'last year' make test TARGET=hw DEVICE=xilinx_u200_xdma_201830_2”. Alternatively, the executable is already present in /nano_rev/old_code/v3 and can be run that way. These executables are also present for the prior versions. To build the other versions just put their source code into the source directory in main and change the makefile to the one in /nano_rev/old_code/v0, as the current one uses an additional file to specify the DDR ports for inputs.


![image](https://user-images.githubusercontent.com/59748092/165870024-1b18a392-3f8c-4afc-90d8-bece7a571106.png)
