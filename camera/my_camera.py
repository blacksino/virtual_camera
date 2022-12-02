import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtk.util import numpy_support
import os
# import vtkmodules.all as vtk
import yaml
import json
from datetime import datetime
from outline_test import *
import time
from rough_registraion.guess_pose import *
import cv2

np.set_printoptions(suppress=True)
import vtk.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtk.vtkRenderingOpenGL2
from vtk.vtkCommonColor import vtkNamedColors
from vtk.vtkFiltersSources import vtkPlaneSource
from vtk.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def loadSTL(filenameSTL):
    readerSTL = vtk.vtkSTLReader()
    readerSTL.SetFileName(filenameSTL)
    # 'update' the reader i.e. read the .stl file
    readerSTL.Update()

    polydata = readerSTL.GetOutput()

    print
    "Number of Cells:", polydata.GetNumberOfCells()
    print
    "Number of Points:", polydata.GetNumberOfPoints()

    # If there are no points in 'vtkPolyData' something went wrong
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("No point data could be loaded from " + filenameSTL)
        return None

    return polydata


def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m


def trans_to_matrix(trans):
    """ Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    matrix = vtk.vtkMatrix4x4()
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            matrix.SetElement(i, j, trans[i, j])
            matrix.SetElement(i, j, trans[i, j])
    return matrix


def setup_background_image(image_data, background_renderer):
    # Set up the background camera to fill the renderer with the image
    # Create an image actor to display the image
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)
    background_renderer.AddActor(image_actor)

    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)


def get_vtk_image_from_numpy(image_in):
    # Copied from this post: http://vtk.1045678.n5.nabble.com/Image-from-OpenCV-as-texture-td5748969.html
    # Thanks to Addison Elliott
    #
    # Use VTK support function to convert Numpy to VTK array
    # The data is reshaped to one long 2D array where the first dimension is the data and the second dimension is
    # the color channels (1, 3 or 4 typically)
    # Note: Fortran memory ordering is required to import the data correctly
    # Type is not specified, default is to use type of Numpy array
    image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    imge_flip = np.transpose(image_rgb, (1, 0, 2))
    imge_flip = np.flip(imge_flip, 1)
    dims = imge_flip.shape
    # print("dims = {}".format(dims))
    size = dims[:2]
    channels = dims[-1]
    vtkArray = numpy_to_vtk(imge_flip.reshape((-1, channels), order='F'), deep=False)

    # Create image, set parameters and import data
    vtk_image = vtk.vtkImageData()

    # For vtkImage, dimensions, spacing and origin is assumed to be 3D. VTK images cannot be larger than 3D and if
    # they are less than 3D, the remaining dimensions should be set to default values. The function padRightMinimum
    # adds default values to make the list 3D.
    vtk_image.SetDimensions(size[0], size[1], 1)
    # vtk_image.SetSpacing(padRightMinimum(self.spacing, 3, 1))
    vtk_image.SetOrigin(0, 0, 0)

    # Import the data (vtkArray) into the vtkImage
    vtk_image.GetPointData().SetScalars(vtkArray)
    return vtk_image



class Camera_VTK:
    """
    Example class showing how to project a 3D world coordinate onto a 2D image plane using VTK
    """

    def __init__(self, w, h, K, mesh_path, data_root_path, background_path=None, read_tet=False):
        self.w = w
        self.h = h
        self.K = K  # Camera matrix

        self.f = np.array([K[0, 0], K[1, 1]])  # focal lengths
        self.c = K[:2, 2]  # principal point

        # projection of 3D sphere center onto the image plane
        if background_path != None:
            self.background_image = cv2.imread(background_path)
            # self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)
            self.background_image = get_vtk_image_from_numpy(self.background_image)
        self.mesh_path = mesh_path
        self.data_path = data_root_path
        # self.init_vtk()
        self.read_tet = read_tet
        self.init_model(self.read_tet)

    def snapshot(self, type='target'):
        cam = self.renderer.GetActiveCamera()
        pixel_points, scene_points = self.iren.GetInteractorStyle().get_points()
        scene_points_normals = self.iren.GetInteractorStyle().get_points_normals()

        # ------ deprecated-----------
        # v_angel = cam.GetViewAngle()
        #
        # rad_v = np.deg2rad(v_angel)
        #
        # wcx, wcy = cam.GetWindowCenter()
        #
        # cx = (1 - wcx) * self.w / 2
        # cy = (1 - wcy) * self.h / 2
        #
        # fx = self.w / (2 * np.tan(rad_h / 2))
        #
        # current_K = np.array([[fx, 0., cx],
        #                       [0., fy, cy],
        #                       [0., 0., 1.]])

        center = self.polyMapper.GetCenter()
        extrinsics = np.linalg.inv(vtkmatrix_to_numpy(cam.GetViewTransformMatrix()))
        """
        vtk中相机的内参由多个option决定，具体可以参阅：vtk.vtkCamera.GetProjectionMatrix的源码。
        对于vtk中最普通，也是默认的归一化视锥体内参，内参中fx，fy由视场角与分辨率大小决定，fy in vtkProjectionMatrix =  fy_original / (h/2)
        fx不能直接设定，而是由指定的像素长宽比设定,i.e.SetUserTransform,因为视场角设定了h与fy,
        这里fy=f(focal_length) * sy(像素大小).具体可以参阅：https://www.jianshu.com/p/935044175ca4
        而cx，cy则是 camera.GetWorldCenter()中归一化的中心wcx与wcy决定。

        总结一下:
        fy = K[1][1] / (h/2)
        fx = fy * aspect(fx与fy之比）= K[0][0] / ( h/2)
        cx = wcx
        cy = wcy   

        """
        intrinsics = vtkmatrix_to_numpy(
            cam.GetProjectionTransformMatrix(1, cam.GetClippingRange()[0], cam.GetClippingRange()[1]))

        """
        这里由于指定了UserTransform所以不需要再次指定aspect ratio,所以长宽比写成1就好。
        """

        K = np.zeros((3, 3))

        K[0][0] = intrinsics[0][0] * (self.h / 2)
        """
        注意这里乘的是self.h
        """
        K[1][1] = intrinsics[1][1] * (self.h / 2)
        K[0][2] = (1 - intrinsics[0][2]) * self.w / 2
        K[1][2] = (1 + intrinsics[1][2]) * self.h / 2

        parameter_dict = dict()
        if type == 'target':
            parameter_dict['extrinsics'] = extrinsics.tolist()
            parameter_dict['intrinsics'] = K.tolist()
        else:
            parameter_dict['extrinsics'] = extrinsics.tolist()
            parameter_dict['intrinsics'] = K.tolist()
            parameter_dict["cam_position_for_openGL"] = cam.GetPosition()
            parameter_dict["look_at_position_for_openGL"] = cam.GetFocalPoint()
            parameter_dict["view_up_for_openGL"] = cam.GetViewUp()
            parameter_dict['cam_projection_matrix'] = vtkmatrix_to_numpy(
                cam.GetProjectionTransformMatrix(1, cam.GetClippingRange()[0], cam.GetClippingRange()[1])).tolist()
            parameter_dict['init_bbox'] = self.polydata.GetBounds()
            parameter_dict['scene_points'] = scene_points
            parameter_dict['scene_points_normals'] = scene_points_normals
            parameter_dict['contour'] = self.iren.GetInteractorStyle().contour
            parameter_dict['contour_index'] = self.iren.GetInteractorStyle().contour_index
        self.axes.SetVisibility(False)
        winToIm = vtk.vtkWindowToImageFilter()
        winToIm.SetInput(self.renWin)
        winToIm.Update()
        vtk_image = winToIm.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = cv2.flip(numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components), 0)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        time = datetime.now().time().strftime("%H:%M:%S")
        os.mkdir(os.path.join(self.data_path, time))
        all_path = os.path.join(self.data_path, time)
        if type == 'target':
            cv2.imwrite(os.path.join(all_path, 'target_frame.png'), arr)
            with open(os.path.join(all_path, 'target_frame.json'), 'w') as f:
                json.dump(parameter_dict, f, indent=1)
        elif type == 'registration':
            cv2.imwrite(os.path.join(all_path, 'registration.png'), arr)
            with open(os.path.join(all_path, 'registration.json'), 'w') as f:
                json.dump(parameter_dict, f, indent=1)
        self.axes.SetVisibility(True)
        print('Image and Parameters has been saved.')

        # now we capture the depth map
        # first we should get the depth buffer
        z_near, z_far = cam.GetClippingRange()
        width, height = self.renWin.GetSize()
        z_buffer_data = vtk.vtkFloatArray()
        self.renWin.GetZbufferData(0, 0, width - 1, height - 1, z_buffer_data)
        z_buffer_data = numpy_support.vtk_to_numpy(z_buffer_data)
        z_buffer_data = z_buffer_data.reshape((height, width))
        # flip z_buffer_data along the y axis
        z_buffer_data = np.flip(z_buffer_data, axis=0)
        # z_buffer is in the range of [0, 1] actually
        z_ndc = 2 * z_buffer_data - 1
        # now we can get the z_eye
        z_eye = 2 * z_near * z_far / (z_ndc * (z_far - z_near) - z_far - z_near)
        # multiply -1 to make the z_eye in the same direction as the z_world
        z_eye = -1 * z_eye
        # mask out the invalid depth
        z_eye[z_buffer_data == 1] = np.nan

        # now rescale the z_eye to the range of [0, 256]
        z_eye = z_eye * 256
        z_eye = z_eye.astype(np.uint8)
        cv2.imwrite(os.path.join(all_path, 'depth.png'), z_eye)
        return

    def init_model(self, read_tet=False):
        # get the file extension name
        if self.mesh_path.endswith('.vtk') or self.mesh_path.endswith('vtu'):
            if self.read_tet:
                if self.mesh_path.endswith('.vtk'):
                    self.mesh_reader = vtk.vtkUnstructuredGridReader()
                elif self.mesh_path.endswith('.vtu'):
                    self.mesh_reader = vtk.vtkXMLUnstructuredGridReader()
                self.mesh_reader.SetFileName(self.mesh_path)
                self.mesh_reader.Update()
                self.filter = vtk.vtkDataSetSurfaceFilter()
                output = self.mesh_reader.GetOutput()
                # convert output to stl format.
                self.filter.SetInputData(self.mesh_reader.GetOutput())
                self.filter.Update()
                self.polydata = self.filter.GetOutput()
            else:
                self.mesh_reader = vtk.vtkPolyDataReader()
                self.mesh_reader.SetFileName(self.mesh_path)
                self.mesh_reader.Update()
                self.polydata = self.mesh_reader.GetOutput()

        else:
            assert self.mesh_path.endswith('.stl'), "Mesh filename must be endswith \'stl\'."
            self.mesh_reader = vtk.vtkSTLReader()
            self.mesh_reader.SetFileName(self.mesh_path)
            self.mesh_reader.Update()
            self.polydata = self.mesh_reader.GetOutput()

        homo = np.array([[-0.58663026, 0.80751273, 0.0615477, -140.99337367],
                         [0.33761624, 0.31292948, -0.88774457, -209.90521739],
                         [-0.73612513, -0.49999833, -0.45620331, 34.01865876],
                         [0., 0., 0., 1.]])

        z_rot = np.array([[0, -1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        homo = z_rot @ np.linalg.inv(homo)
        tfm = vtk.vtkTransform()
        tfm.SetMatrix(trans_to_matrix(homo))
        filter = vtk.vtkTransformFilter()
        filter.SetInputData(self.polydata)
        filter.SetTransform(tfm)
        filter.Update()
        self.polydata = filter.GetOutput()
        self.polyMapper = vtk.vtkPolyDataMapper()
        self.polyMapper.SetInputData(self.polydata)
        # self.polyMapper.SetInputConnection(self.stl_reader.GetOutputPort())

        """
        set mapper

        """
        self.meshActor = vtk.vtkActor()
        self.meshActor.SetMapper(self.polyMapper)

        self.axes = vtk.vtkAxesActor()
        self.axes.SetTotalLength(10000, 10000, 10000)

        # planeSource = vtkPlaneSource()
        # planeSource.SetCenter(0.0, -10, 0.0)
        # planeSource.SetOrigin(-1000.0, -10, -1000.0)
        # planeSource.SetPoint1(-1000.0, -10, 1000.0)
        # planeSource.SetPoint2(1000.0, -10, -1000.0)
        # planeSource.Update()
        #
        # plane = planeSource.GetOutput()
        #
        #
        # # Create a mapper and actor
        # mapper = vtkPolyDataMapper()
        # mapper.SetInputData(plane)

        colors = vtkNamedColors()

        # Set the background color.
        colors.SetColor('BkgColor', [26, 51, 77, 255])
        # actor = vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(colors.GetColor3d('Banana'))
        # actor.GetProperty().SetLighting(False)
        self.background_render = vtk.vtkRenderer()
        self.background_render.SetLayer(0)
        self.background_render.InteractiveOff()
        setup_background_image(self.background_image,self.background_render)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetLayer(1)
        self.renderer.AddActor(self.meshActor)
        self.renderer.AddActor(self.axes)
        # self.renderer.AddActor(actor)
        self.axes.SetVisibility(True)
        self.renderer.SetBackground(0, 0, 0)

        self.camera = vtk.vtkCamera()
        aspect = self.f[1] / self.f[0]
        v_angle = 180 / np.pi * 2.0 * np.arctan2(self.h / 2.0, self.f[1])
        wcx = -2.0 * (self.c[0] - self.w / 2.0) / self.w
        wcy = 2.0 * (self.c[1] - self.h / 2.0) / self.h

        """
        TODO:
        垃圾的vtk由于不能显式的指定投影矩阵,这里存在一个当wcx不为0时的bug。

        """
        self.camera.SetWindowCenter(wcx, wcy)
        self.camera.SetViewAngle(v_angle)

        m = np.eye(4)
        m[0, 0] = 1.0 / aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        self.camera.SetUserTransform(t)

        self.renderer.SetActiveCamera(self.camera)
        # self.camera.GetFrustumPlanes()

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetNumberOfLayers(2)
        self.renWin.SetSize(self.w, self.h)
        if self.h == 1080 and self.w == 1920:
            self.renWin.SetFullScreen(True)

        self.renWin.AddRenderer(self.renderer)
        self.renWin.AddRenderer(self.background_render)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        self.style = MyInteractor(self.polydata)
        self.style.picker = vtk.vtkCellPicker()
        self.style.picker.AddPickList(self.meshActor)
        self.style.SetDefaultRenderer(self.renderer)
        self.style.AddObserver('RightButtonPressEvent', self.style.RightButtonPressEvent)
        self.iren.SetInteractorStyle(self.style)
        self.iren.AddObserver('KeyPressEvent', self.key_press_call_back)

        self.iren.Initialize()
        self.iren.Start()

    def key_press_call_back(self, obj, en, ):
        key = self.iren.GetKeySym()
        if key == 'space':
            if len(self.iren.GetInteractorStyle().scene_points) == 0:
                print("This frame has been frozen,Notice That there is no initial points,we assume that this frame is "
                      "target frame,which record target extrinsics.")
                self.snapshot(type='target')
            elif len(self.iren.GetInteractorStyle().scene_points) >= 4 and len(
                    self.iren.GetInteractorStyle().contour) >= 0:
                print(
                    f"This frame has been frozen,Notice That there are {len(self.iren.GetInteractorStyle().scene_points)} initial "
                    f"points,{len(self.iren.GetInteractorStyle().contour)} contour points, we assume that this frame is "
                    "to record label information in 3D")
                self.snapshot(type='registration')
            else:
                print(
                    f"Not enough initial registration points and contour,there are  {len(self.iren.GetInteractorStyle().scene_points)} "
                    f"initial points,{len(self.iren.GetInteractorStyle().contour)} contour points.\n")
                print("Label More!")
                return
        elif key == '1':
            self.iren.GetInteractorStyle().label_points = True
            self.iren.GetInteractorStyle().label_contour = False
            print("Start Labeling points.")
        elif key == '2':
            self.iren.GetInteractorStyle().label_points = False
            self.iren.GetInteractorStyle().label_contour = True
            print("Start Labeling contour.")
        elif key == 'k':
            self.iren.GetInteractorStyle().pred_point_id = -1
            self.iren.GetInteractorStyle().current_point_id = -1
            print("current contour labeling has been done,start another one.")
        elif key == 'l':
            self.iren.GetInteractorStyle().label_points = False
            self.iren.GetInteractorStyle().label_contour = False
            print("Exit Labeling mode.")
        elif key == '8':
            current_opacity = self.meshActor.GetProperty().GetOpacity()
            if current_opacity > 1 or current_opacity < 0:
                return
            else:
                print("Decrease Opacity")
                self.meshActor.GetProperty().SetOpacity(current_opacity - 0.1)
                self.iren.GetRenderWindow().Render()

        elif key == '9':
            current_opacity = self.meshActor.GetProperty().GetOpacity()
            if current_opacity > 1:
                return
            else:
                print("Increase Opacity")
                self.meshActor.GetProperty().SetOpacity(current_opacity + 0.1)
                self.iren.GetRenderWindow().Render()
        elif key == 's':
            print('Extracting Silhouette.')
            self.silhouette = vtk.vtkPolyDataSilhouette()
            self.silhouette.SetInputData(self.iren.GetInteractorStyle().mesh)
            self.silhouette.SetCamera(self.renderer.GetActiveCamera())
            # self.silhouette.SetEnableFeatureAngle(1)
            self.silhouette.SetFeatureAngle(0)
            self.silhouette.SetEnableFeatureAngle(0)
            self.silhouette.BorderEdgesOn()
            self.silhouette.GetBorderEdges()
            self.sil_mapper = vtk.vtkPolyDataMapper()
            self.sil_mapper.SetInputConnection(self.silhouette.GetOutputPort())
            self.silhouette.Update()
            self.sil_actor = vtk.vtkActor()
            self.sil_actor.SetMapper(self.sil_mapper)
            self.sil_actor.GetProperty().SetColor(0.1, 0.4, 1)
            self.sil_actor.GetProperty().SetLineWidth(3)
            self.renderer.AddActor(self.sil_actor)


        elif key == 'd':
            print('Reset Camera.')
            self.camera.SetPosition(0, 0, 1)
            self.camera.SetFocalPoint(0, 0, 0)
            # flush the pipeline
            self.iren.GetRenderWindow().Render()

        elif key == 'c':
            print('Clear Silhouette.')
            self.renderer.RemoveActor(self.sil_actor)
            self.renderer.AddActor(self.meshActor)
            self.renderer.Render()
        elif key == 't':
            self.renderer.ResetCamera()
            self.renderer.Render()
            print('Generate Random Pose Guess via last record.')
            data = read_json('/data/endoscope/simulation_data/10:11:42/registration.json')
            look_at = data['look_at_position_for_openGL']
            origin = data['cam_position_for_openGL']

            look_at = np.array(look_at)
            origin = np.array(origin)

            look_at_direction = (look_at - origin) / np.linalg.norm(look_at - origin)

            bbox = data['init_bbox']
            bbox = np.array(bbox)
            cam_projection_matrix = data['cam_projection_matrix']
            cam_projection_matrix = np.array(cam_projection_matrix)
            K = data['intrinsics']
            K = np.array(K)
            znear, zfar = determine_znear_zfar(bbox, cam_projection_matrix, K)

            assert znear < 0 and zfar < 0, "znear and zfar should be negative"
            assert znear > zfar, "znear should be greater than zfar"

            bbox_center = np.array([(bbox[1] + bbox[0]) / 2, (bbox[3] + bbox[2]) / 2, (bbox[5] + bbox[4]) / 2])

            z_distance_range = znear
            random_position = z_distance_range * look_at_direction + origin
            translation = random_position - bbox_center

            # now we rotate the object
            all_normals = data['scene_points_normals']
            all_normals = np.array(all_normals)
            main_direction = get_main_direction_of_normals(all_normals)

            # rotate the object
            rotation_matrix = get_rotation_matrix(main_direction, look_at_direction)

            # generate homogeneous transformation matrix
            homogeneous_transformation_matrix = np.zeros((4, 4))
            # homogeneous_transformation_matrix[0:3, 0:3] = rotation_matrix
            homogeneous_transformation_matrix[0:3, 3] = translation
            homogeneous_transformation_matrix[3, 3] = 1
            # homogeneous_transformation_matrix = np.linalg.inv(homogeneous_transformation_matrix)
            # tfm = vtk.vtkMatrix4x4()
            # for i in range(4):
            #     for j in range(4):
            #         tfm.SetElement(i, j, homogeneous_transformation_matrix[i, j])
            self.camera.SetPosition(origin)
            self.camera.SetFocalPoint(look_at)
            self.iren.GetRenderWindow().Render()
        return


class MyInteractor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, mesh_data, parent=None):

        self.collection_point = []
        self.PointCount = 0
        self.time_count = 0
        self.mesh = mesh_data
        self.pixel_points = []
        self.scene_points = []
        self.scene_points_normals = []
        self.pred_point_id = -1
        self.current_point_id = -1
        self.label_points = False
        self.label_contour = False
        self.contour = []
        self.contour_index = []

    def OnRightButtonUp(self):
        return

    def RightButtonPressEvent(self, obj, en):
        if not self.label_contour and not self.label_points:
            return
        clickPos = self.GetInteractor().GetEventPosition()
        print("Picking pixel: ", clickPos)

        # Pick from this location
        picker = self.picker
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        cell_id = picker.GetCellId()
        if cell_id != -1:
            # coords = vtk.vtkCoordinate()
            # coords.SetCoordinateSystemToDisplay()
            # coords.SetValue(clickPos[0],clickPos[1],0)
            # worldcoords = coords.GetComputedWorldValue(self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer());
            # If CellId = -1, nothing was picked
            point_position = picker.GetPickPosition()
            # get normal vector of the picked point
            normal = picker.GetPickNormal()

            print("Pick position is: ", point_position)
            print("Picked cell is: ", cell_id)
            print("cell related vertex id :", self.mesh.GetCell(cell_id).GetPointId(0),
                  self.mesh.GetCell(cell_id).GetPointId(1), self.mesh.GetCell(cell_id).GetPointId(2))

            if self.label_points:
                self.pixel_points.append(clickPos)
                self.scene_points.append(point_position)
                self.scene_points_normals.append(normal)
            elif self.label_contour:
                self.current_point_id = picker.GetPointId()
                if self.pred_point_id != -1 and self.label_contour:
                    self.draw_lines()
                self.pred_point_id = self.current_point_id

            # Create a sphere
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(point_position)
            # sphereSource.SetRadius(0.2)
            sphereSource.SetRadius(1)

            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if self.label_points:
                actor.GetProperty().SetColor(1.0, 0.0, 0.0)
            elif self.label_contour:
                actor.GetProperty().SetColor(0.0, 0.0, 1.0)
            self.GetDefaultRenderer().AddActor(actor)

        # Forward events
        self.OnRightButtonDown()
        return

    def get_points(self):
        return self.pixel_points, self.scene_points

    def get_points_normals(self):
        return self.scene_points_normals

    def draw_lines(self):
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self.mesh)
        dijkstra.SetStartVertex(self.pred_point_id)
        dijkstra.SetEndVertex(self.current_point_id)
        dijkstra.Update()

        idList = dijkstra.GetIdList()
        points = vtk.vtkPoints()
        seg_index = list()
        seg_position = list()
        for i in range(idList.GetNumberOfIds() - 1, -1, -1):
            current_id = idList.GetId(i)
            seg_index.append(current_id)
            seg_position.append(self.mesh.GetPoint(current_id))
            points.InsertNextPoint(self.mesh.GetPoint(current_id))

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
        for j in range(points.GetNumberOfPoints()):
            polyline.GetPointIds().SetId(j, j)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(cells)
        render = self.GetDefaultRenderer()
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(polyData)
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetColor(0, 1, 0)
        lineActor.GetProperty().SetLineWidth(3)
        # for k in range(points.GetNumberOfPoints()-1,-1,-1):
        #     current_contour_seg.append(points.GetPoint(k))
        #     current_contour_index.append(idList.GetId(k))
        self.contour.append(seg_position)
        self.contour_index.append(seg_index)
        render.AddActor(lineActor)
        render.Render()
