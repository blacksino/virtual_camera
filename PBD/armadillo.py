import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from render_tools import *
from math_tools import *

import pypbd as pbd
import numpy as np


# 1 = distance constraints (PBD)
# 2 = FEM tet constraints (PBD)
# 3 = strain tet constraints (PBD)
# 4 = shape matching
# 5 = distance constraints (XPBD)
simModel = 2


def buildModel():   
    sim = pbd.Simulation.getCurrent()
    
    # Init simulation and collision detection with standard settings
    sim.initDefault()
    model = sim.getModel()
    
    rbs = model.getRigidBodies()
    
    # Create the mesh for the solid model
    createMesh(simModel)
       
    # Load geometry from OBJ file
    fileName2 = "./data/models/cube.obj"
    (vdTorus, meshTorus) = pbd.OBJLoader.loadObjToMesh(fileName2, [1, 1, 1])
    fileName3 = "./data/models/sphere.obj"
    (vdSphere,meshSphere) = pbd.OBJLoader.loadObjToMesh(fileName3,[1,1,1])

    # ground
    rb = model.addRigidBody(1.0,            # density
        vdTorus, meshTorus,                 # vertices, mesh
        [0.0, 0, 0],                    # position
        np.identity(3),                     # rotation matrix
        [100, 2, 100],                    # scaling
        testMesh=True,                      # test mesh vertices against other signed distance fields
        generateCollisionObject= True,      # automatically generate an SDF and a corresponding collision object
        resolution=[50, 50, 50])            # resolution of the signed distance field
    rb.setMass(0.0)
    rb.setFrictionCoeff(0.85)
    rb.setRestitutionCoeff(0.001)

    rb_1 = model.addRigidBody(1.0,            # density
        vdSphere, meshSphere,                 # vertices, mesh
        [0, 100, 0],                    # position
        np.identity(3),                     # rotation matrix
        [0.01, 0.01, 0.01],                    # scaling
        testMesh=True,                      # test mesh vertices against other signed distance fields
        generateCollisionObject= True,      # automatically generate an SDF and a corresponding collision object
        resolution=[30, 30, 30])
    rb_1.setMass(1.0)

    # Set collision tolerance (distance threshold)
    cd = sim.getTimeStep().getCollisionDetection()      
    cd.setTolerance(0.00001)
    
    ts = sim.getTimeStep()
    ts.setValueUInt(pbd.TimeStepController.NUM_SUB_STEPS, 5)

     
# Create a particle model mesh 
def createMesh(simModel):
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    # Load TetGen model from file
    fileNameNode = '/home/SENSETIME/xulixin2/Downloads/001713343/simple_liver.node'
    fileNameEle = '/home/SENSETIME/xulixin2/Downloads/001713343/simple_liver.1.ele'

    (positions, tets) = pbd.TetGenLoader.loadTetgenModel(fileNameNode, fileNameEle)

    # move up and scale vertices
    pos = np.array(positions)
    for x in pos:
        x[0] =  x[0]/25 - 10
        x[1] = x[1]/25
        x[2] = x[2]/ 25 -20
    # Generata tetrahedra model
    tetModel = model.addTetModel(pos, tets, testMesh=True)

    tetModel.setRestitutionCoeff(0.001)
    tetModel.setFrictionCoeff(0.2)
    # init constraints
    stiffness = 100
    if (simModel == 5):
        stiffness = 100000
    poissonRatio = 0.49
    
    # Generate a default set of constraints for the tet model to simulate stretching and shearing resistance
    # simModel: 
    # 1 = distance constraints (PBD)
    # 2 = FEM tet constraints (PBD)
    # 3 = strain tet constraints (PBD)
    # 4 = shape matching
    # 5 = distance constraints (XPBD)
    model.addSolidConstraints(tetModel, simModel, stiffness, poissonRatio, 1, False, False)

    # 这里tetmodel是我们添加的tetrahedron
    print("Number of tets: " + str(tetModel.getParticleMesh().numTets()))
    print("Number of vertices: " + str(len(pos)))
 
# Render all bodies     
def render():
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    # render meshes of rigid bodies
    rbs = model.getRigidBodies()   
    for rb in rbs:
        vd = rb.getGeometry().getVertexData()
        mesh = rb.getGeometry().getMesh()
        drawMesh(vd, mesh, 0, [0,0.2,0.7])
        
    # render mesh of tet model     
    pd = model.getParticles()   
    tetModel = model.getTetModels()[0]
    offset = tetModel.getIndexOffset()
    drawMesh(pd, tetModel.getSurfaceMesh(), offset, [0,0.2,0.7])
    
    # render time
    glPushMatrix()
    glLoadIdentity()
    drawText([-0.95,0.9], "Time: {:.2f}".format(pbd.TimeManager.getCurrent().getTime()))
    glPopMatrix()
    
# Perform simulation steps
def timeStep():
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    # We perform 8 simulation steps per render step
    for i in range(2):
        sim.getTimeStep().step(model)
        
    # Update mesh normals of tet model for rendering
    for tetModel in model.getTetModels():
        tetModel.updateMeshNormals(model.getParticles())
     
# Reset the simulation
def reset():
    sim = pbd.Simulation.getCurrent()
    sim.reset()
    sim.getModel().cleanup()
    cd = sim.getTimeStep().getCollisionDetection()      
    cd.cleanup()
    buildModel()
    
def main():
    # Activate logger to output info on the console
    pbd.Logger.addConsoleSink(pbd.LogLevel.INFO)
    
    # init OpenGL
    initGL(1280, 1024)  
    gluLookAt (0, 30, -30, 0, 0, 0, 0, 1, 0)
    glRotatef(20.0, 0, 1, 0)
    
    # build simulation model
    buildModel()

    # start event loop 
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset()
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glClearColor(0.3, 0.3, 0.3, 1.0)
        timeStep()
        render()
        pygame.display.flip()
        
    pygame.quit()
    pbd.Timing.printAverageTimes()

if __name__ == "__main__":
    main()