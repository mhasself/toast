
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_FFT_HPP
#define TOAST_FFT_HPP

#include <vector>


namespace toast {
enum class fft_plan_type {
    fast,
    best
};

enum class fft_direction {
    forward,
    backward
};


// This uses aligned memory allocation
typedef std::vector <double, toast::simd_allocator <double> > fft_data;

class FFTPlanReal1D {
    public:

        typedef std::shared_ptr <FFTPlanReal1D> pshr;

        static FFTPlanReal1D * create(int64_t length, int64_t n,
                                      fft_plan_type type,
                                      fft_direction dir, double scale);

        virtual ~FFTPlanReal1D() {}

        virtual void exec() {
            return;
        }

        virtual std::vector <double *> tdata() {
            return std::vector <double *> ();
        }

        virtual std::vector <double *> fdata() {
            return std::vector <double *> ();
        }

        int64_t length();

        int64_t count();

    protected:

        FFTPlanReal1D(int64_t length, int64_t n, fft_plan_type type,
                      fft_direction dir, double scale);

        int64_t length_;
        int64_t n_;
        double scale_;
        fft_plan_type type_;
        fft_direction dir_;
};


// R1D FFT plan store

class FFTPlanReal1DStore {
    public:

        ~FFTPlanReal1DStore();
        static FFTPlanReal1DStore & get();
        void cache(int64_t len, int64_t n = 1);
        FFTPlanReal1D::pshr forward(int64_t len, int64_t n = 1);
        FFTPlanReal1D::pshr backward(int64_t len, int64_t n = 1);
        void clear();

    private:

        FFTPlanReal1DStore() {}

        std::map <std::pair <int64_t, int64_t>, FFTPlanReal1D::pshr> fplans_;
        std::map <std::pair <int64_t, int64_t>, FFTPlanReal1D::pshr> rplans_;
};
}

#endif // ifndef TOAST_RNG_HPP
