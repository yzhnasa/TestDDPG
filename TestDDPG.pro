TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG += no_keywords
#CONFIG -= qt

SOURCES += \
        main.cpp

HEADERS += \
    environment.h \
    model.h \
    reinforcement_learning.h \
    utilities.h

INCLUDEPATH += -I "C:\Qt\libtorch\include"
INCLUDEPATH += -I "C:\Qt\libtorch\include\torch\csrc\api\include"
INCLUDEPATH += -I "C:\Program Files (x86)\Eigen3\include\eigen3"

LIBS += -LC:\Qt\libtorch\lib
LIBS += C:\Qt\libtorch\lib\*.lib
#LIBS += C:\Qt\libtorch\lib\*.dll

LIBS += -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64"
LIBS += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64\*.lib"

unix {
QMAKE_LFLAGS += -Wl,--no-as-needed
}
win32 {
QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ
QMAKE_LFLAGS += -INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z
QMAKE_LFLAGS += /machine:x64
}
