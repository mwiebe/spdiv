#include <Python.h>
#include <numpy/arrayobject.h>

static PyArrayObject*
SpecialDivide(PyArrayObject* a, PyArrayObject* b, PyArrayObject *out)
{
    NpyIter *iter = NULL;
    PyArrayObject *op[3];
    PyArray_Descr *dtypes[3];
    npy_uint32 flags, op_flags[3];

    /* Iterator construction parameters */
    op[0] = a;
    op[1] = b;
    op[2] = out;

    dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    if (dtypes[0] == NULL) {
        return NULL;
    }
    dtypes[1] = dtypes[0];
    dtypes[2] = dtypes[0];

    flags = NPY_ITER_BUFFERED |
            NPY_ITER_EXTERNAL_LOOP |
            NPY_ITER_GROWINNER |
            NPY_ITER_REFS_OK |
            NPY_ITER_ZEROSIZE_OK;

    /* Every operand gets the flag NPY_ITER_USE_MASKNA */
    op_flags[0] = NPY_ITER_READONLY |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_USE_MASKNA;
    op_flags[1] = op_flags[0];
    op_flags[2] = NPY_ITER_WRITEONLY |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_USE_MASKNA |
                  NPY_ITER_NO_BROADCAST |
                  NPY_ITER_ALLOCATE;

    iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER,
                            NPY_SAME_KIND_CASTING, op_flags, dtypes);
    /* Don't need the dtype reference anymore */
    Py_DECREF(dtypes[0]);
    if (iter == NULL) {
        return NULL;
    }
    if (NpyIter_GetIterSize(iter) > 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *stridesptr, *countptr;

        /* Variables needed for looping */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return NULL;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        stridesptr = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);
        do {
            /* Data pointers and strides needed for innermost loop */
            char *data_a = dataptr[0], *data_b = dataptr[1];
            char *data_out = dataptr[2];
            char *maskna_a = dataptr[3], *maskna_b = dataptr[4];
            char *maskna_out = dataptr[5];
            npy_intp stride_a = stridesptr[0], stride_b = stridesptr[1];
            npy_intp stride_out = stridesptr[2];
            npy_intp maskna_stride_a = stridesptr[3];
            npy_intp maskna_stride_b = stridesptr[4];
            npy_intp maskna_stride_out = stridesptr[5];
            npy_intp i, count = *countptr;

            for (i = 0; i < count; ++i) {
                /* If neither of the inputs are NA */
                if (NpyMaskValue_IsExposed((npy_mask)*maskna_a) &&
                            NpyMaskValue_IsExposed((npy_mask)*maskna_b)) {
                    double a_val = *(double *)data_a;
                    double b_val = *(double *)data_b;
                    /* Do the divide if 'b' isn't zero */
                    if (b_val != 0.0) {
                        *(double *)data_out = a_val / b_val;
                        /* Need to also set this element to exposed */
                        *maskna_out = NpyMaskValue_Create(1, 0);
                    }
                    /* Otherwise output an NA without touching its data */
                    else {
                        *maskna_out = NpyMaskValue_Create(0, 0);
                    }
                }
                /* Turn the output into NA without touching its data */
                else {
                    *maskna_out = NpyMaskValue_Create(0, 0);
                }

                data_a += stride_a;
                data_b += stride_b;
                data_out += stride_out;
                maskna_a += maskna_stride_a;
                maskna_b += maskna_stride_b;
                maskna_out += maskna_stride_out;
            }
        } while (iternext(iter));
    }
    if (out == NULL) {
        out = NpyIter_GetOperandArray(iter)[2];
    }
    Py_INCREF(out);
    NpyIter_Deallocate(iter);

    return out;
}


static PyObject *
spdiv(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *a, *b, *out = NULL;
    static char *kwlist[] = {"a", "b", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O&", kwlist, 
                            &PyArray_AllowNAConverter, &a,
                            &PyArray_AllowNAConverter, &b,
                            &PyArray_OutputAllowNAConverter, &out)) {
        return NULL;
    }

    /*
     * The usual NumPy way is to only use PyArray_Return when
     * the 'out' parameter is not provided.
     */
    if (out == NULL) {
        return PyArray_Return(SpecialDivide(a, b, out));
    }
    else {
        return (PyObject *)SpecialDivide(a, b, out);
    }
}

static PyMethodDef SpDivMethods[] = {
    {"spdiv", (PyCFunction)spdiv, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initspdiv_mod(void)
{
    PyObject *m;

    m = Py_InitModule("spdiv_mod", SpDivMethods);
    if (m == NULL) {
        return;
    }

    /* Make sure NumPy is initialized */
    import_array();
}
