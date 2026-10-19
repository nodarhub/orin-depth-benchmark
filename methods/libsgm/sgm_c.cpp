// C interface to sgm::StereoSGM for ctypes (8-bit grey input, 16-bit subpixel output, host pointers).
#include <libsgm.h>

extern "C" {

void* sgm_create(int width, int height, int disp_size, int paths8, int subpixel, int P1, int P2)
{
	sgm::StereoSGM::Parameters p;
	p.P1 = P1;
	p.P2 = P2;  // libSGM defaults otherwise (P1=10, P2=120 unless set), uniqueness 0.95, LR_max_diff 1, symmetric census 9x7
	p.subpixel = subpixel != 0;
	p.path_type = paths8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
	return new sgm::StereoSGM(width, height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST, p);
}

void sgm_execute(void* h, const unsigned char* left, const unsigned char* right, unsigned short* dst)
{
	static_cast<sgm::StereoSGM*>(h)->execute(left, right, dst);
}

int sgm_invalid(void* h) { return static_cast<sgm::StereoSGM*>(h)->get_invalid_disparity(); }

void sgm_destroy(void* h) { delete static_cast<sgm::StereoSGM*>(h); }
}
