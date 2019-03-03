#ifndef _EIGEN3_HDF5_EXTEND
#define _EIGEN3_HDF5_EXTEND

#include <typeinfo>
#include <string>
#include <vector>

#include <H5Cpp.h>
#include <Eigen/Dense>

#include "eigen3-hdf5.hpp"

namespace EigenHDF5
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Function to copy vector<Vector> into MatrixX
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace internal_math
    {
        template<typename Scalar> bool convert_vectorVectorX_to_MatrixX(const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vec_of_vecX, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &MatX)
        {
            // size of vector
            int nvector_size = vec_of_vecX.size();

            // error check
            if (nvector_size == 0) { return false; }

            // size of each VectorXd
            int nVectorX_size = vec_of_vecX[0].size();

            //allocate size
            MatX.resize(nVectorX_size, nvector_size);


            // Loop and copy to MatriX*
            for (int i = 0; i < nvector_size; i++)
            {

                // error check
                if (vec_of_vecX[i].size() != nVectorX_size)
                {
                    MatX.resize(0, 0);
                    return false;
                }

                // copy to MatrixXd
                MatX.col(i) = vec_of_vecX[i];
            }

            return true;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // native std::vector all except std<string>
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T, typename A> void save(H5::CommonFG &h5group, const std::string &name, const std::vector<T, A> &var)
    {
        // Part 2: set dimension to 1 (single value)
        hsize_t dim[1];
        dim[0] = var.size();

        // Part 3: create the type
        H5::DataSpace space(1, dim);

        // Part 4:
        if (typeid(T) == typeid(int))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_INT32, space));
            dataset.write(var.data(), H5::PredType::NATIVE_INT32);
        }
        else if (typeid(T) == typeid(float))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_FLOAT, space));
            dataset.write(var.data(), H5::PredType::NATIVE_FLOAT);
        }
        else if (typeid(T) == typeid(double))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space));
            dataset.write(var.data(), H5::PredType::NATIVE_DOUBLE);
        }
        else if (typeid(T) == typeid(char))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_CHAR, space));
            dataset.write(var.data(), H5::PredType::NATIVE_CHAR);
        }
        else
        {
            //not implemented
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // This function saves a vector of variable length string to hdf5
    // see: http://stackoverflow.com/questions/581209/how-to-best-write-out-a-stdvector-stdstring-container-to-a-hdf5-dataset
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void save(H5::CommonFG &h5group, const std::string &name, const std::vector<std::string> &var)
    {
        // Part 1: grab pointers to the chars
        std::vector<const char*> chars;
        for (const auto& str : var) {
            chars.push_back(str.data());
        }

        // Part 2: get the dim, rank = 1 (1D array)
        hsize_t dim[1];
        dim[0] = chars.size();

        // Part 3: create the type
        H5::DataSpace space(1, dim);
        auto s_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);

        // Part 4: write the output to a scalar dataset
        H5::DataSet dataset(h5group.createDataSet(name, s_type, space));
        dataset.write(chars.data(), s_type);
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // std::vector<Eigen::VectorX*>
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename Scalar>
    void save(H5::CommonFG &h5group, const std::string &name, const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vec_of_vecX)
    {
        // Part 1: create MatrixX* to store final results
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatX;

        // Part 2: copy vector<VectorX> into MatrixX
        internal_math::convert_vectorVectorX_to_MatrixX(vec_of_vecX, MatX);

        // Part 3: save MatrixX using Eigenhdf5
        save(h5group, name, MatX);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // save "simple" data (int/float/bool)
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T> void save_simple(H5::CommonFG &h5group, const std::string &name, const T &var)
    {
        // Part 2: set dimension to 1 (single value)
        hsize_t dim[1];
        dim[0] = 1;

        // Part 3: create the type
        H5::DataSpace space(1, dim);

        // Part 4:
        if (typeid(T) == typeid(int))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_INT32, space));
            dataset.write(&var, H5::PredType::NATIVE_INT32);
        }
        else if (typeid(T) == typeid(float))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_FLOAT, space));
            dataset.write(&var, H5::PredType::NATIVE_FLOAT);
        }
        else if (typeid(T) == typeid(double))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space));
            dataset.write(&var, H5::PredType::NATIVE_DOUBLE);
        }
        else if (typeid(T) == typeid(char))
        {
            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_CHAR, space));
            dataset.write(&var, H5::PredType::NATIVE_CHAR);
        }
        else if (typeid(T) == typeid(bool))
        {
            // cast to int since hdf5 doesnt support boolean 
            int int_from_bool = int(var);

            H5::DataSet dataset(h5group.createDataSet(name, H5::PredType::NATIVE_INT32, space));
            dataset.write(&int_from_bool, H5::PredType::NATIVE_INT32);
        }
        else
        {
            //not implemented
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // This function saves a vector of variable length string to hdf5
    // see: http://stackoverflow.com/questions/581209/how-to-best-write-out-a-stdvector-stdstring-container-to-a-hdf5-dataset
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void save_simple(H5::CommonFG &h5group, const std::string &name, const std::string &var)
    {
        // Part 1: grab pointers to the chars
        const char *c = var.c_str();

        // Part 2: get the dim, rank = 1 (1D array)
        hsize_t dim[1];
        dim[0] = 1;

        // Part 3: create the type
        H5::DataSpace space(1, dim);
        auto s_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);

        // Part 4: write the output to a scalar dataset
        H5::DataSet dataset(h5group.createDataSet(name, s_type, space));
        dataset.write(&c, s_type);
    }
}

#endif