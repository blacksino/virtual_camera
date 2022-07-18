import numpy as np
import SimpleITK as sitk
import vtk
# from vtk.util import numpy_support
from vtkmodules.util import numpy_support
from scipy.ndimage import binary_fill_holes


def readimg(path):
    sitk_image = sitk.ReadImage(path)
    return sitk_image


def SitkToVTK(sitk_image, label):
    voldims = np.asarray(sitk_image.GetSize())
    print(sitk_image.GetSize())
    npvol = sitk.GetArrayFromImage(sitk_image)
    npvol = (npvol == label).astype(np.uint8)
    npvol = npvol.reshape(np.prod(voldims))
    vtkimg = vtk.vtkImageData()
    vtkimg.SetDimensions(voldims)
    vtkimg.SetExtent([0,voldims[0]-1, 0,voldims[1]-1, 0,voldims[2]-1])
    vtkimg.SetSpacing(np.asarray(sitk_image.GetSpacing()))
    vtkimg.SetOrigin(np.asarray(sitk_image.GetOrigin()))

    # Get a VTK wrapper around the numpy volume
    dataName = 'MRI_STACK'
    vtkarr = numpy_support.numpy_to_vtk(npvol)
    vtkarr.SetName(dataName)
    vtkimg.GetPointData().AddArray(vtkarr)
    vtkimg.GetPointData().SetScalars(vtkarr)
    return vtkimg


def save_vtk(polydata, path):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(path)
    writer.Update()


def teeth_MarchingCubes(image):
    # mc = vtk.vtkMarchingCubes()
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, 1.0)
    mc.SetValue(1, 2.0)
    mc.SetValue(2, 3.0)
    mc.Update()
    meshdata = mc.GetOutput()
    return meshdata


def Smooth_stl(stl):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(stl)
    smoothFilter.SetNumberOfIterations(50)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    smoothMeshData = smoothFilter.GetOutput()
    return smoothMeshData


def MeshDataColorFill(meshdata):
    cellN = meshdata.GetNumberOfCells()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    for i in range(cellN):
        if meshdata.GetCellData().GetScalars().GetTuple(i)[0] == 1:
            colors.InsertNextTuple((255, 0, 0))
        elif meshdata.GetCellData().GetScalars().GetTuple(i)[0] == 2:
            colors.InsertNextTuple((0, 0, 255))
        elif meshdata.GetCellData().GetScalars().GetTuple(i)[0] == 3:
            colors.InsertNextTuple((0, 255, 0))
        else:
            colors.InsertNextTuple((0, 128, 128))
    meshdata.GetCellData().SetScalars(colors)
    segMeshMapper = vtk.vtkPolyDataMapper()
    segMeshMapper.SetInputData(meshdata)
    return segMeshMapper


def polydata2itkimg(poly, sitk_image):
    dims = np.asarray(sitk_image.GetSize())
    spaces = np.asarray(sitk_image.GetSpacing())
    origins = np.asarray(sitk_image.GetOrigin())

    pts = poly.GetPoints()
    pt_array = pts.GetData()
    pt_array = numpy_support.vtk_to_numpy(pt_array)
    pt_array_xyz = ((pt_array - origins) / spaces + 0.5).astype(np.uint)
    imgarr = np.zeros((dims[2], dims[1], dims[0]))
    imgarr[pt_array_xyz[:, 2], pt_array_xyz[:, 1], pt_array_xyz[:, 0]] = 1
    for i in range(imgarr.shape[0]):
        imgarr[i] = binary_fill_holes(imgarr[i] > 0).astype(np.uint8)
    print(imgarr.shape)
    newsitkimg = sitk.GetImageFromArray(imgarr)
    newsitkimg.CopyInformation(sitk_image)
    sitk.WriteImage(newsitkimg, "/home/yuxiang/Data/debug/tmp/debug.nii.gz")


def generate_stl(niifile, stlfile, label):
    sitk_image = readimg(niifile)
    vtkimg = SitkToVTK(sitk_image, label)
    polydata = teeth_MarchingCubes(vtkimg)
    polydata = Smooth_stl(polydata)
    # polydata = MeshDataColorFill(polydata)
    save_vtk(polydata, stlfile)
    # polydata2itkimg(polydata, sitk_image)


if __name__ == '__main__':
    # artery_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500003.4894755618831920366-liver_vessel_seg_art.nii.gz'
    # artery_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/artery.stl'
    # generate_stl(artery_nii, artery_stl, 40)
    #
    # vessel_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500008.5143273006649809924-liver_vessel_seg.nii.gz'
    # for lab, lab_name in enumerate('hv,pv,ivc,bile_duct'.split(',')):
    #     vessel_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/{}.stl'.format(lab_name)
    #     generate_stl(vessel_nii, vessel_stl, lab+1)
    #
    # bone_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500008.5143273006649809924-abdomen_bone_seg.nii.gz'
    # for lab, lab_name in enumerate('rib,vertebrae'.split(',')):
    #     bone_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/{}.stl'.format(lab_name)
    #     generate_stl(bone_nii, bone_stl, lab+1)

    organ_dict = {
        1: 'spleen',
        2: 'right_kidney',
        3: 'left_kidney',
        4: 'gallbladder',
        5: 'liver',
        6: 'stomach',
        10: 'duodenum',
        11: 'pancreas',
        13: 'other_bowel'
    }

    organ_nii_path = '/data/4DCT/1 Bai Xining/S3000_organ.nii.gz'
    organ_stl_path = f'/data/4DCT/1 Bai Xining/S3000_liver.stl'
    generate_stl(organ_nii_path,organ_stl_path,5)


    # organ_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500008.5143273006649809924-liver_organ_seg.nii.gz'
    # for lab, lab_name in organ_dict.items():
    #     organ_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/{}.stl'.format(lab_name)
    #     generate_stl(organ_nii, organ_stl, lab)
    #
    # segment_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500008.5143273006649809924-liver_segment_seg.nii.gz'
    # for lab in range(1, 9):
    #     segment_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/liver_segment_{}.stl'.format(lab)
    #     generate_stl(segment_nii, segment_stl, lab)
    #
    # lesion_nii = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/1.3.46.670589.33.1.63731094351315318500008.5143273006649809924_lesion_mask.nii.gz'
    # lesion_arr = sitk.GetArrayFromImage(sitk.ReadImage(lesion_nii))
    # lesion_labs = np.unique(lesion_arr)[1:]
    # for lab in lesion_labs:
    #     lesion_stl = '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343/liver_lesion_{}.stl'.format(lab)
    #     generate_stl(lesion_nii, lesion_stl, lab)

    # import os
    # from medai_base_data_define.rle import rle2numpy
    #
    # if not os.path.exists('/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343'):
    #     os.makedirs('/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343')
    #
    # for r, fo, fl in os.walk('/data2/huaxi_art_pv_data/vessel_to_stl_example/1.3.46.670589.33.1.63731094138370138800001.5270764413352239469'):
    #     for f in fl:
    #         if f.endswith('.rle'):
    #             rle_str = np.fromfile(os.path.join(r, f), dtype=np.int32)
    #             # with open(os.path.join(r, f), 'rb') as fi:
    #             #     rle_str = fi.read()
    #             flatten_arr = rle2numpy(rle_str)
    #             seg_arr = flatten_arr.reshape([-1, 512, 512])
    #             seg_itk = sitk.GetImageFromArray(seg_arr)
    #             sitk.WriteImage(seg_itk, os.path.join(os.path.join(
    #                 '/data2/huaxi_art_pv_data/vessel_to_stl_example/001713343', f.replace('.rle', '.nii.gz')
    #             )))

    # vesself = '/data2/huaxi_art_pv_data/origin/artmask/hcc_0000703802_PA0_ST1_SE4_art.nii.gz'
    # stlf = '/data2/huaxi_art_pv_data/origin/artmask/hcc_0000703802_PA0_ST1_SE4_art.stl'
    # generate_stl(vesself, stlf, 5)
    # organf = "/data/CT_Product_Liver_Vessel_Prediction/package_result_v4/1696681_4309065_1.25mm_stnd_c+_2_0_0_axial/Organ/" \
    #          "1696681_4309065_1.25mm_stnd_c+_2_0_0_axial.nii_0_organ.nii.gz"
    # stlf = "/data/CT_Product_Liver_Vessel_Prediction/package_result_v4/1696681_4309065_1.25mm_stnd_c+_2_0_0_axial/Organ/%s.stl"
    # for index, organ in enumerate(["spleen", "kidney_r", "kidney_l", "gallbladder", "liver", "stomach", "IVC", "aorta"]):
    #     generate_stl(organf, stlf % organ, index+1)
    #
    # segmentsf = "/data/CT_Product_Liver_Vessel_Prediction/package_result_v4/1696681_4309065_1.25mm_stnd_c+_2_0_0_axial/Vessel/" \
    #           "1696681_4309065_1.25mm_stnd_c+_2_0_0_axial.nii_segments0.nii.gz"
    # for label, organ in zip([1, 2, 3, 4, 5, 6, 7, 8, 9], ["liver-S1", "liver-S2", "liver-S3", "liver-S4", "liver-S5", "liver-S6",
    #                                                       "liver-S7", "liver-S8", "liver-S9"]):
    #     generate_stl(segmentsf, stlf % organ, label)
    #
    # vesselsf = "/data/CT_Product_Liver_Vessel_Prediction/package_result_v4/1696681_4309065_1.25mm_stnd_c+_2_0_0_axial/Vessel/" \
    #            "1696681_4309065_1.25mm_stnd_c+_2_0_0_axial.nii_vessel0.nii.gz"
    # for label, organ in zip([1, 2], ["hepatic_vein", "portal_vein"]):
    #     generate_stl(vesselsf, stlf % organ, label)
    #
    # lesionsf = "/data/CT_Product_Liver_Vessel_Prediction/package_result_v4/1696681_4309065_1.25mm_stnd_c+_2_0_0_axial/Lesion/" \
    #            "1696681_4309065_1.25mm_stnd_c+_2_0_0_axial.nii_lesion0.nii.gz"
    # for label, organ in zip([1], ["lesion"]):
    #     generate_stl(lesionsf, stlf % organ, label)