from PySide6.QtWidgets import QMainWindow, QLabel, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QSpinBox, QCheckBox, QFileDialog
from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon
import numpy as np
import nibabel as nib
import sys
import os

BASEDIR = os.path.dirname(__file__)
try:
    from ctypes import windll  # Only exists on Windows.
    myappid = 'mycompany.myproduct.subproduct.version'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass


class ViewBlock(QWidget):
    def __init__(self, slice_shape, n_slices, obj, axis=0):
        super().__init__()
        self.frame = 0
        self.slice_shape = slice_shape
        self.obj = obj
        self.axis = axis
        self.rendered_labels = []
        self.indicator = False 


        # planar_pixmap view
        self.block_layout = QVBoxLayout()
        # pixmap
        self.pixmap = QLabel()
        self.pixmap.setGeometry(QRect(0,0,slice_shape[0], slice_shape[1]))
        self.pixmap.setScaledContents(True)
        self.pixmap.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.pixmap.setMaximumSize(QSize(400,400))
        img = QPixmap(slice_shape[0],slice_shape[1])
        img.fill(Qt.black)
        self.pixmap.setPixmap(img)
        # spinbox
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(n_slices-1)
        self.spinbox.setSingleStep(1) 
        self.spinbox.valueChanged.connect(self.spinbox_update)
        # slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(n_slices-1)
        self.slider.setSingleStep(1)
        self.slider.sliderMoved.connect(self.slider_update)
        # frame indicies
        self.indicies_box = QCheckBox("Slice indicies")
        self.indicies_box.stateChanged.connect(self.indicies_box_state)
        # slider-spinbox block
        self.controller_layout = QHBoxLayout()
        self.controller_layout.addWidget(self.slider)
        self.controller_layout.addWidget(self.spinbox)
        self.controller_layout.addWidget(self.indicies_box)
        # label checkboxes
        self.label_checkboxes = QVBoxLayout()
        self.core = QCheckBox("Tumor core")
        self.enh = QCheckBox("GD Enhancing")
        self.edema = QCheckBox("Edema")
        checkboxes = [self.core, self.enh, self.edema] 
        for QW in checkboxes:
            self.label_checkboxes.addWidget(QW)
        self.core.setObjectName("core")
        self.enh.setObjectName("enh")
        self.edema.setObjectName("edema")
        self.core.stateChanged.connect(self.core_)
        self.enh.stateChanged.connect(self.enh_)
        self.edema.stateChanged.connect(self.edema_)

        
        # whole block
        self.block_layout.addWidget(self.pixmap)
        self.block_layout.addLayout(self.controller_layout)
        self.block_layout.addLayout(self.label_checkboxes)
        
        self.setLayout(self.block_layout)

    def core_(self):
        if self.core.isChecked():
            self.rendered_labels.append(0)
            self.core.setStyleSheet("#core {background-color:red}")
        else:
            self.rendered_labels.remove(0)
            self.core.setStyleSheet("#core {background-color:white}")
        self.update_slice()

    def enh_(self):
        if self.enh.isChecked():
            self.rendered_labels.append(2)
            self.enh.setStyleSheet("#enh {background-color:green}")
        else:
            self.rendered_labels.remove(2)
            self.enh.setStyleSheet("#enh {background-color:white}")
        self.update_slice()

    def edema_(self):
        if self.edema.isChecked():
            self.rendered_labels.append(1)
            self.edema.setStyleSheet("#edema {background-color:yellow}")
        else:
            self.rendered_labels.remove(1)
            self.edema.setStyleSheet("#edema {background-color:white}")
        self.update_slice()
    
    def indicies_box_state(self):
        if self.indicies_box.isChecked():
            self.indicator = True
            self.update_slice()
        else:
            self.indicator = False
            self.update_slice()


    def slider_update(self):
        self.frame = self.slider.value()
        self.spinbox.setValue(self.frame)
        self.update_slice()
    
    def spinbox_update(self):
        self.frame = self.spinbox.value()
        self.slider.setValue(self.frame)
        self.update_slice()

    def update_slice(self):
        self.slice_ = self.obj.slice_data(axis=self.axis, frame=self.frame, labels=self.rendered_labels)
        if self.indicator:
            self.update_indicator(self.axis)
        else:
            img = QImage(self.slice_, self.slice_shape[0], self.slice_shape[1], QImage.Format_RGB888)
            self.pixmap.setPixmap(QPixmap(img))

    def update_indicator(self,axis):
        if not hasattr(self, "slice_"): return 0
        copy = self.slice_.copy()
        if self.indicator:
            if axis == 0:
                copy[154-self.obj.planar_block.frame,::4,:2] = 255
                copy[::4,239-self.obj.coronal_block.frame,:2] = 255
            elif axis == 1:
                copy[154-self.obj.planar_block.frame,::4,:2] = 255
                copy[::4,239-self.obj.sagital_block.frame,:2] = 255
            else:
                copy[239-self.obj.coronal_block.frame,::4,:2] = 255
                copy[::4,239-self.obj.sagital_block.frame,:2] = 255


        img = QImage(copy, self.slice_shape[0], self.slice_shape[1], QImage.Format_RGB888)
        self.pixmap.setPixmap(QPixmap(img))




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MRI slicer") 
        self.setFixedSize(QSize(1000,400))
        self.layout = QHBoxLayout()
        self.planar_block = ViewBlock((240,240),155, obj=self, axis=2)
        self.sagital_block = ViewBlock((240,155),240,obj=self, axis=0)
        self.coronal_block = ViewBlock((240,155),240, obj=self, axis=1)
        self.layout.addWidget(self.planar_block)
        self.layout.addWidget(self.sagital_block)
        self.layout.addWidget(self.coronal_block)

        # menu
        self.load_sample_action = QAction(QIcon(os.path.join(BASEDIR,"folder.ico")), "&Load sample", self)
        self.load_sample_action.setStatusTip("Load data")
        self.load_sample_action.setCheckable(True)
        self.load_sample_action.triggered.connect(self.load_sample)

        self.load_target_action = QAction(QIcon(os.path.join(BASEDIR,"folder.ico")), "&Load target", self)
        self.load_target_action.setStatusTip("Load data")
        self.load_target_action.setCheckable(True)
        self.load_target_action.triggered.connect(self.load_target)

        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&Data")
        self.file_menu.addAction(self.load_sample_action)
        self.file_menu.addAction(self.load_target_action)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        # default data
        self.data = np.zeros((240,240,155))
        self.label_data = [np.stack([np.zeros((240,240,155)), np.zeros((240,240,155)), np.zeros((240,240,155))]),
                           np.stack([np.zeros((240,240,155)), np.zeros((240,240,155)), np.zeros((240,240,155))]),
                           np.stack([np.zeros((240,240,155)), np.zeros((240,240,155)), np.zeros((240,240,155))])]


    def unify_shapes(self, slice_):
        slice_ = np.transpose(slice_)
        slice_ = np.rot90(np.rot90(slice_))
        return slice_
    
    def unify_data(self, x):
        if len(x.shape) == 4:
            x = np.argmax(x, axis=-1)
        if x.shape != (240,240,155):
            data = np.zeros((240,240,155))
            data[56:184,56:184,13:141] = x
            x = data
            del data
        return x

    def slice_data(self, axis, frame, labels=None):
        labels = sorted(labels)
        # 2 - G - enh
        # 1 - Y - edema
        # 0 - R - core
        if axis == 0:  # sagital
            slice_ = self.data[frame, :,:]
            slice_ = np.stack([slice_, slice_, slice_])
            for i in labels:
                slice_ += self.label_data[i][:,frame, :,:]
            slice_ = self.unify_shapes(slice_)
            
            # update slice indc.
            self.planar_block.update_indicator(2)
            self.coronal_block.update_indicator(1)

        elif axis == 1: # coronal
            slice_ = self.data[:,frame,:]
            slice_ = np.stack([slice_, slice_, slice_])
            for i in labels:
                slice_ += self.label_data[i][:,:,frame,:]
            slice_ = self.unify_shapes(slice_)
            
            # update slice indc.
            self.planar_block.update_indicator(2)
            self.sagital_block.update_indicator(0)

        elif axis == 2: # planar
            slice_ = self.data[:,:,frame]
            slice_ = np.stack([slice_, slice_, slice_])
            for i in labels:
                slice_ += self.label_data[i][:,:,:,frame]
            slice_ = self.unify_shapes(slice_)

            # update slice indc.
            self.sagital_block.update_indicator(0)
            self.coronal_block.update_indicator(1)

        else:   
            raise ValueError(f"{axis} is not supported")

        slice_ = np.clip(slice_, 0, 255)
        slice_ = np.require(slice_, np.uint8, 'C')
        return slice_
    

    def fileName(self):
        filename = QFileDialog.getOpenFileName(self, "Load data", os.getcwd(), "*.nii *.tiff *.tif *.npy")
        return filename[0]

    def update_frames(self):
        self.planar_block.update_slice()
        self.sagital_block.update_slice()
        self.coronal_block.update_slice()

    def load_target(self):
        path = self.fileName()
        if isinstance(path, str):
            if path.endswith('.nii'):
                self.target = self.load_nii(path)
            elif path.endswith('.npy'):
                self.target = self.load_npy(path)
            else:
                raise TypeError(f"This file type is not supported: {path}")
            self.target = self.unify_data(self.target)
            core = (self.target == 1)*100
            edema = (self.target == 2)*100
            enh = (self.target == 4)*100
            self.label_data = [np.stack([core, np.zeros((240,240,155)), np.zeros((240,240,155))]),
                            np.stack([edema, edema, np.zeros((240,240,155))]),
                            np.stack([np.zeros((240,240,155)), enh, np.zeros((240,240,155))])]
            self.update_frames()
            del self.target
        
    def load_sample(self):
        path = self.fileName()
        if isinstance(path, str):
            if path.endswith('.nii'):
                self.target = self.load_nii(path)
            elif path.endswith('.npy'):
                self.target = self.load_npy(path)
            else:
                raise TypeError(f"This file type is not supported: {path}")
            self.data = nib.load(path).get_fdata()
            self.data = (self.data/np.max(self.data))*255
            self.update_frames()
    
    def load_npy(self, path):
        return np.load(path) 

    def load_nii(self, path):
        return nib.load(path).get_fdata()


if __name__ == '__main__':
    app = QApplication()
    app.setWindowIcon(QIcon(os.path.join(BASEDIR,"brain--plus.ico"))) 
    window = MainWindow()
    window.show()
    app.exec()



