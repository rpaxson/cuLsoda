# CuLsoda
### Overview
CuLsoda is a port from fortran of the Livermore Solver for Ordinary Differential Equations with Automatic Switching and Double Precision (DLSODA) to nVidia's CUDA framework.  

### Current State
CuLsoda is the result of a summer internship back in 2009, and after being brought into a barely functional state as proof of concept, the internship ended and the code hasn't been touched since.  As I've been contacted by several individuals interested in the code over the last few years, I am posting it to GitHub in the hopes that it will be picked up and that development will recommence.

##### Tasks
* Get rid of GOTO statements.
* Optimize for CUDA.
* Comment Everything.
* Possibly port to CUDA C++?
* Branch and create an OpenCL version.

#### Files
This repository currently includes three subfolders in the Benchmarks folder. Each of these represents a benchmark that I ran for my paper.  The main code is taken from the multigpu folder and it is assumed (perhaps unwisely) that this is the most recent and feature complete version of the code.  To my knowledge, if properly compiled, the code in the individual benchmark folders ought to work properly.

### Contact
Anyone with questions is welcome to contact me at Celemourn@&lt;insert popular free email provider whose name is five letters, begins with G and ends with L here&gt;.com