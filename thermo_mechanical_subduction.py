
# coding: utf-8

# ## 2D thermo mechanical subduction models
# 
# This notebook develops a simple, flexible model for 2d subduction using Underworld2. All functionality is parallel-compatible. The code has been run on 48 processors, at a vertical resolution of 256 elements (Q1).
# 
# 
# <hr>
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is copyright Dan Sandiford and Louis Moresi. It is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

# ## Notes
# 
# ### General
# 
# Many of the functions and scripts (unsupported module) were written by Dan Sandiford during his PhD, and have minimal safeguards. When things break (they will), post and issue, and we'll try to provide a fix!
# 
# ### Checkpointing
# 
# All parts of the code that relate to checkpointing have the following identifier:
# 
# `#*************CHECKPOINT-BLOCK**************#`
# 
# ### Materials
# 
# Currently a weak material is added to the top of the lower plate to decouple the plates. For simplicity, in this notebook, there is no functionality for updating the distribution of materials (i.e. creating new crust). An example of one way you could add this functionality can be found here: https://github.com/dansand/materialTransformations
# 
# ### Solver
# 
# We have found the penalty method works well for this problem. This is the default setting. (Unfortunately, it doesn't scale well to 3D.)
# 
# ### Resolution and particles
# 
# Properly resolving the weak crust is critical to this model, which means that resolution (and particle numbers) need to be quite high (i.e. don't expect quick models on your laptop). The default crust thickness is 15 km. By switching the depth of the model to something like 660 km (upper mantle), you might get away with running at a resolution of ~ 72 processors. Note that the dyanmics are quite sensitive to the depth of the model. If you are looking to do more than just tinker, it will be worthwhile refining the mesh in the vertical direction. 
# 
# 
# ## nn_evaluation
# 
# In a couple of places a function called nn_evaluation (nearest neighbour evaluation) is called. This is used whenever we want to map a swarm variable to a mesh variable.

# ## Package requirements
# 
# Assuming you're running underworld2 through a docker image, you will need the following python packages:
# 
# * easydict  https://pypi.python.org/pypi/easydict/
# 
# It should be simple enough to do `pip install easydict` (though you'll need to do this for each container you start)
# 
# You'll also need a copy of this module: https://github.com/dansand/unsupported.git
# 
# This is occasionally merged with the official underworld2 module of the same name, but it's safest to grab a copy of the repo: 
# 
# `git clone https://github.com/dansand/unsupported.git`, 
# 
# (easiest to clone straight into in the same directory as this notebook.)
# 
# To add the module to the pythonpath do:
# 
# ```python
# import os
# import sys
# 
# if os.getcwd() == '/workspace/newSlab':
#     sys.path.append('./unsupported')
#     
# ```

# In[58]:

import os
import sys

if '/workspace' in os.getcwd(): #a simple test for the standard Docker image
    #sys.path.append('./unsupported') #unsupported module located in current working dir
    sys.path.append('../unsupported') #unsupported module located in parent dir.

else:
    pass #put your non-docker path here


# In[59]:

import numpy as np
import underworld as uw
import math
from underworld import function as fn
import glucifer
from easydict import EasyDict as edict
import operator
import pickle


#
from unsupported_dan.utilities.interpolation import nn_evaluation
from unsupported_dan.utilities.subduction import slab_top
from unsupported_dan.interfaces.marker2D import markerLine2D
from unsupported_dan.easymodels import checkpoint
from unsupported_dan.easymodels import easy_args
from unsupported_dan.utilities.misc import *


# ## Setup output directories

# In[60]:

###########
#Standard output directory setup
###########


#Model letter identifier demarker
Model = "T"

#Model number identifier demarker:
ModNum = 1

#Any isolated letter / integer command line args are interpreted as Model/ModelNum and read in here
#We use this for doing parameter sweeps

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then assume it's a not a parameter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg
                
                
###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
#checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
xdmfPath = outputPath + 'xdmf/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    if not os.path.isdir(xdmfPath):
        os.makedirs(xdmfPath)
        
uw.barrier() #Barrier here so no procs run the check in the next cell too early        


# ## Checkpointing

# In[61]:

#*************CHECKPOINT-BLOCK**************#

#cp = checkpoint(outputPath + 'checkpoint/', loadpath='./results/A/1/checkpoint/10')
cp = checkpoint(outputPath + 'checkpoint/')

#*************CHECKPOINT-BLOCK**************#


# ## Model parameters and scaling
# 
# The model is controlled by 4 dictionaries, 
# 
# * 'dp': dimensional parameters
# * 'md': model dictionary (changing numerics, switching certain processes on / off)
# * 'sf': scaling factors
# * 'ndp': non dimensional parameters
# 
# The dictionaries use the easyDict format, so that the following format can be applied to items:
# 
# ```
# 
# key.value = blah
# 
# ```
# 
# This approach was chosen so that:
# 
# 1. the non-dimensionalisation process is competely clear in each model
# 2. parameters can be easy saved (dictionaries can be 'pickled')
# 3. parameters can be easily altered through command line args. 
# 
# Command line arguments can only be provided to the `dp` and `md` dictionaries. The logic is that the scaling process (non-dimesionalisation) should basically be static, and simply provides a mapping between physical units and model units.

# In[62]:

dp = edict({})
#Main physical paramters
dp.depth=1000e3                         #model Depth (width set by md.aspectRatio)
dp.refDensity=3300.                     #reference density
dp.refGravity=9.8                       #surface gravity
dp.viscosityScale=1e20                  #reference upper mantle visc., 
dp.refDiffusivity=1e-6                  #thermal diffusivity
dp.refExpansivity=3e-5                  #surface thermal expansivity
dp.gasConstant=8.314                    #gas constant
dp.specificHeat=1250.                   #Specific heat (Jkg-1K-1)
dp.potentialTemp=1573.                  #mantle potential temp (K)
dp.surfaceTemp=273.                     #surface temp (K)
#Rheology - flow law paramters
dp.cohesionMantle=20e6                  #mantle cohesion in Byerlee law
dp.cohesionCrust=1e6                    #crust cohesion in Byerlee law
dp.frictionMantle=0.2                   #mantle friction coefficient in Byerlee law (tan(phi))
dp.frictionCrust=0.02                   #crust friction coefficient 
dp.diffusionPreExp=5.34e-10             #1./1.87e9, pre-exp factor for diffusion creep
dp.diffusionEnergy=3e5 
dp.diffusionVolume=5e-6
dp.lowerMantlePreExp=4.23e-15           
dp.lowerMantleEnergy=2.0e5
dp.lowerMantleVolume=1.5e-6
dp.lowerMantleViscFac = 30.
#Rheology - cutoff values
dp.viscosityMin=1e18
dp.viscosityMax=1e25                    #viscosity max in the mantle material
dp.viscosityMinCrust=1e20               #viscosity min in the weak-crust material
dp.viscosityMaxCrust=1e20               #viscosity max in the weak-crust material
dp.yieldStressMax=300*1e6              
dp.crustViscCutoffDepth = 100e3
dp.crustViscEndWidth = 20e3
#Intrinsic Lengths
dp.crustThickness = 15.*1e3             #weak layer thickness on top of slabs
dp.crustMantleDepth=250.*1e3 
dp.lowerMantleDepth=660.*1e3  
dp.crustLimitDepth=650.*1e3             #Deeper than this, crust material rheology reverts to mantle rheology
#Slab and plate init. parameters
dp.subZoneLoc=-100e3                    #X position of subduction zone...km
dp.leftRidge=-1.*(5000e3)               
dp.rightRidge=(5000e3)
dp.maxDepth=150e3
dp.theta=40                             #Angle of slab
dp.radiusOfCurv = 250e3                 #radius of curvature
dp.slabMaxAge=70e6                      #age of subduction plate at trench
dp.plateMaxAge=100e6                    #max age of slab (Plate model)
dp.opMaxAge=35e6                        #age of op
#Misc
dp.stickyAirDepth=100e3                 #depth of sticky air layer
dp.viscosityStickyAir=1e19              #stick air viscosity, normal
#derived params
dp.deltaTemp = dp.potentialTemp-dp.surfaceTemp



#Modelling and Physics switches

md = edict({})
md.refineMeshStatic=True
md.stickyAir=False
md.aspectRatio=5.
md.res=32
md.ppc=25                               #particles per cell
md.elementType="Q1/dQ0"                 #"Q2/DPC1"
md.secInvFac=math.sqrt(1.)
md.courantFac=0.5                       #extra limitation on timestepping may need to be decreased if using "Q2/DPC1" )
md.thermal = True                       #thermal system or compositional
md.swarmInitialFac = 0.6                #initial swarm layout will be int(md.ppc*md.swarmInitialFac), popControl will densify later
md.compBuoyancy = False
md.nltol = 0.01
md.maxSteps = 1000
md.checkpointEvery = 50
md.swarmUpdate = 10
md.penaltyMethod = True
md.opuniform = False
md.spuniform = False
md.opfixed = False
md.spfixed = False
md.buoyancyFac = 1.0
#time-based actions
md.filesFreqYears = 1.0e6 #dimensional time in years



uw.barrier()


# In[63]:

##Parse any command-line args

sysArgs = sys.argv

#We want to run this on both the parameter dict, and the model dict
easy_args(sysArgs, dp)
easy_args(sysArgs, md)


uw.barrier()


# In[64]:

sf = edict({})

sf.lengthScale=2900e3
sf.viscosityScale = dp.viscosityScale
sf.stress = (dp.refDiffusivity*sf.viscosityScale)/sf.lengthScale**2
#sf.lithGrad = dp.refDensity*dp.refGravity*(sf.lengthScale)**3/(sf.viscosityScale*dp.refDiffusivity) 
sf.lithGrad = (sf.viscosityScale*dp.refDiffusivity) /(dp.refDensity*dp.refGravity*(sf.lengthScale)**3)
sf.velocity = dp.refDiffusivity/sf.lengthScale
sf.strainRate = dp.refDiffusivity/(sf.lengthScale**2)
sf.time = 1./sf.strainRate
sf.actVolume = (dp.gasConstant*dp.deltaTemp)/(dp.refDensity*dp.refGravity*sf.lengthScale)
sf.actEnergy = (dp.gasConstant*dp.deltaTemp)
sf.diffusionPreExp = 1./sf.viscosityScale
sf.deltaTemp  = dp.deltaTemp
sf.pressureDepthGrad = (dp.refDensity*dp.refGravity*sf.lengthScale**3)/(dp.viscosityScale*dp.refDiffusivity)


#dimesionless params
ndp  = edict({})
ndp.rayleigh = md.buoyancyFac*(dp.refExpansivity*dp.refDensity*dp.refGravity*dp.deltaTemp*sf.lengthScale**3)/(dp.viscosityScale*dp.refDiffusivity)
ndp.dissipation = (dp.refExpansivity*sf.lengthScale*dp.refGravity)/dp.specificHeat
ndp.surfaceTemp = dp.surfaceTemp/sf.deltaTemp  #Ts
ndp.potentialTemp = dp.potentialTemp/sf.deltaTemp - ndp.surfaceTemp #Tp' = Tp - TS
#lengths / distances
ndp.depth = dp.depth/sf.lengthScale
ndp.leftLim = -0.5*ndp.depth*md.aspectRatio
ndp.rightLim = 0.5*ndp.depth*md.aspectRatio
ndp.crustThickness = dp.crustThickness/sf.lengthScale
ndp.leftRidge = max(ndp.leftLim,  dp.leftRidge/sf.lengthScale)
ndp.rightRidge = min(ndp.rightLim, dp.rightRidge/sf.lengthScale)
ndp.crustLimitDepth = dp.crustLimitDepth/sf.lengthScale
ndp.lowerMantleDepth = dp.lowerMantleDepth/sf.lengthScale
#times - for convenience and sanity the dimensional values are in years, conversion to seconds happens here
ndp.slabMaxAge =  dp.slabMaxAge*(3600*24*365)/sf.time
ndp.plateMaxAge =  dp.plateMaxAge*(3600*24*365)/sf.time
ndp.opMaxAge = dp.opMaxAge*(3600*24*365)/sf.time
#Rheology - flow law paramters
ndp.cohesionMantle=dp.cohesionMantle/sf.stress                  #mantle cohesion in Byerlee law
ndp.cohesionCrust=dp.cohesionCrust/sf.stress                  #crust cohesion in Byerlee law
ndp.frictionMantle=dp.frictionMantle/sf.lithGrad                  #mantle friction coefficient in Byerlee law (tan(phi))
ndp.frictionCrust=dp.frictionCrust/sf.lithGrad                  #crust friction coefficient 
ndp.diffusionPreExp=dp.diffusionPreExp/sf.diffusionPreExp                #pre-exp factor for diffusion creep
ndp.diffusionEnergy=dp.diffusionEnergy/sf.actEnergy
ndp.diffusionVolume=dp.diffusionVolume/sf.actVolume
ndp.lowerMantlePreExp=dp.lowerMantlePreExp/sf.diffusionPreExp 
ndp.lowerMantleEnergy=dp.lowerMantleEnergy/sf.actEnergy
ndp.lowerMantleVolume=dp.lowerMantleVolume/sf.actVolume
ndp.yieldStressMax=dp.yieldStressMax/sf.stress 
#Rheology - cutoff values
ndp.viscosityMin= dp.viscosityMin /sf.viscosityScale
ndp.viscosityMax=dp.viscosityMax/sf.viscosityScale
ndp.viscosityMinCrust= dp.viscosityMinCrust /sf.viscosityScale
ndp.viscosityMaxCrust = dp.viscosityMaxCrust/sf.viscosityScale
ndp.lowerMantleViscFac = dp.lowerMantleViscFac
ndp.crustViscCutoffDepth = dp.crustViscCutoffDepth/sf.lengthScale
ndp.crustViscEndWidth = dp.crustViscEndWidth/sf.lengthScale
#Slab and plate init. parameters
ndp.subZoneLoc = dp.subZoneLoc/sf.lengthScale
ndp.maxDepth = dp.maxDepth/sf.lengthScale
ndp.radiusOfCurv = dp.radiusOfCurv/sf.lengthScale


# In[65]:

#*************CHECKPOINT-BLOCK**************#


#if restart, attempt to read in saved dicts. 
if cp.restart:
    try:
        with open(os.path.join(cp.loadpath, 'dp.pkl'), 'rb') as fp:
                            dp = pickle.load(fp)
        with open(os.path.join(cp.loadpath, 'sf.pkl'), 'rb') as fp:
                            sf = pickle.load(fp)
        with open(os.path.join(cp.loadpath, 'md.pkl'), 'rb') as fp:
                            md = pickle.load(fp)

    except:
        print("couldn't load paramter dictionaries on restart")


    
#add dicts to the checkpointinng object
cp.addDict(dp, 'dp')
cp.addDict(sf, 'sf')
cp.addDict(md, 'md')

#*************CHECKPOINT-BLOCK**************#


# ## Build Mesh and FE variables

# In[66]:

#Domain and Mesh paramters
yres = int(md.res)
xres = int(md.res*12) 



mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres), 
                                 minCoord    = (ndp.leftLim, 1. - ndp.depth), 
                                 maxCoord    = (ndp.rightLim, 1.)) 

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

if md.thermal:
    temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 ) #create this only if Adv-diff
    diffusivityFn = fn.misc.constant(1.)
    
    
    
# Any extra mesh vars. we want to define (mostly to facilite saving as xdmf)
strainRateField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
viscosityField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
    


# In[67]:

#*************CHECKPOINT-BLOCK**************#
cp.addObject(velocityField,'velocityField')
cp.addObject(pressureField,'pressureField')
if md.thermal:
    cp.addObject(temperatureField,'temperatureField')
    cp.addObject(temperatureDotField,'temperatureDotField')
    

#*************CHECKPOINT-BLOCK**************#


# In[68]:

#print(cp.objDict.keys())


# In[69]:

#*************CHECKPOINT-BLOCK**************#

if cp.restart:
    velocityField.load(cp.loadpath + '/velocityField.h5')
    pressureField.load(cp.loadpath + '/pressureField.h5')
    if md.thermal:
        temperatureField.load(cp.loadpath + '/temperatureField.h5')
        temperatureDotField.load(cp.loadpath + '/temperatureDotField.h5')
        
#*************CHECKPOINT-BLOCK**************#


# In[70]:

#miscellaneous Uw functions functions

coordinate = fn.input()
depthFn = mesh.maxCoord[1] - coordinate[1] #a function providing the depth


xFn = coordinate[0]  #a function providing the x-coordinate
yFn = coordinate[1]





#Create a binary circle
def inCircleFnGenerator(centre, radius):
    coord = fn.input()
    offsetFn = coord - centre
    return fn.math.dot( offsetFn, offsetFn ) < radius**2


# ## Boundary Conditions

# In[71]:

#Stokes BCs

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls =mesh.specialSets["MinJ_VertexSet"]
      
        
freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( iWalls, jWalls) )


# In[72]:

#Energy BCs

if md.thermal:
    dirichTempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              indexSetsPerDof=(tWalls,) )


# ## Swarm

# In[73]:

#Materials
mantleID = 0
crustID = 1
airID = 2      #in case we use sticky air

#list of all material indexes
material_list = [mantleID, crustID, airID]


# In[74]:

#*************CHECKPOINT-BLOCK**************#

swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
cp.addObject(swarm,'swarm')
materialVariable      = swarm.add_variable( dataType="int", count=1 )
cp.addObject(materialVariable,'materialVariable')
proxyTempVariable = swarm.add_variable( dataType="double", count=1 )
if not md.thermal:
    cp.addObject(proxyTempVariable,'proxyTempVariable')


if cp.restart:
    swarm.load(cp.loadpath + '/swarm.h5')
    materialVariable.load(cp.loadpath + '/materialVariable.h5')
    if not md.thermal:
        proxyTempVariable.load(cp.loadpath + '/proxyTempVariable.h5')   


else:
    layout = uw.swarm.layouts.PerCellRandomLayout(swarm=swarm, particlesPerCell=int(md.ppc*md.swarmInitialFac))
    swarm.populate_using_layout( layout=layout ) # Now use it to populate.
    proxyTempVariable.data[:] = 0.0
    materialVariable.data[:] = mantleID

#*************CHECKPOINT-BLOCK**************#


# In[75]:

#These variables don;t need checkpointing. They can / should be rebuilt

signedDistanceVariable = swarm.add_variable( dataType="double", count=1 )
#directorVector   = swarm.add_variable( dataType="double", count=2)

#directorVector.data[:,:] = 0.0
signedDistanceVariable.data[:] = 0.0


# In[76]:


#Pass this to Figures to see full extent
bBox=((mesh.minCoord[0], mesh.minCoord[1]),(mesh.maxCoord[0], mesh.maxCoord[1]))


# ## Initial Conditions

# In[77]:

#plate depth

#Flags to control plate behaviour: md.opuniform, md.spuniform, md.opfixed, md.spfixed

#T&S 4.126
thicknessAtTrench = 2.32*math.sqrt(1.*ndp.slabMaxAge)

sig = 150e3/sf.lengthScale
ridgeFn = 1. -                  fn.math.exp(-1.*(xFn - ndp.leftRidge)**2/(2 * sig**2))-                 fn.math.exp(-1.*(xFn - ndp.rightRidge)**2/(2 * sig**2))

spAge = ndp.slabMaxAge*fn.math.abs((ndp.leftRidge - xFn)/(ndp.subZoneLoc - ndp.leftRidge))
opAge = ndp.opMaxAge*fn.math.abs((ndp.rightRidge - xFn)/(ndp.subZoneLoc - ndp.rightRidge))

if md.spuniform:
    if not md.spfixed:
        spAge = ridgeFn*ndp.slabMaxAge
    else:
        spAge = fn.misc.constant(ndp.slabMaxAge)
        
if md.opuniform:
    if not md.opfixed:
        opAge = ridgeFn*ndp.opMaxAge
    else:
        opAge = fn.misc.constant(ndp.opMaxAge)

proxyageFn = fn.branching.conditional([(xFn <= ndp.subZoneLoc, spAge), #idea is to make this arbitrarily complex
                                  (True, opAge)])



# ## Marker lines  for slab, fault, tracking

# ### slab top

# In[78]:

help(slab_top)


# In[79]:

#Create some slab gradient functions to use with slab_top()


def linearGradientFn(S):
    return np.tan(np.deg2rad(-45.))


def circGradientFn(S):
    if S == 0.:
        return 0.
    elif S < ndp.radiusOfCurv:
        return -S/np.sqrt((ndp.radiusOfCurv**2 - S**2))
    else:
        return -1e5
    
    
def polyGradientFn(S):
    if S == 0.:
        return 0.
    else:
        return -1*(S/ndp.radiusOfCurv)**2


# In[80]:

ds = 5e3/sf.lengthScale
normal = [1.,0.]



#data = slab_top([ndp.subZoneLoc, 1.0], normal, linearGradientFn, ds, ndp.maxDepth, mesh)
data = slab_top([ndp.subZoneLoc, 1.0], normal, polyGradientFn, ds, ndp.maxDepth, mesh)
#data = slab_top([ndp.subZoneLoc, 1.0], normal, circGradientFn, ds, ndp.maxDepth, mesh)


# In[81]:

slabxs = data[:,0]
slabys = data[:,1]


# In[82]:

slabLine = markerLine2D(mesh, velocityField, slabxs, slabys, thicknessAtTrench, 1.)


# In[83]:

#Assign the signed distance for the slab - in this case we only want the portion where the signed distance is positive
sd, pts = slabLine.compute_signed_distance(swarm.particleCoordinates.data, distance=2.*thicknessAtTrench)
signedDistanceVariable.data[np.logical_and(sd>0, sd<=slabLine.thickness)] = sd[np.logical_and(sd>0, sd<=slabLine.thickness)]


#Note distance=2.*thicknessAtTrench: we actually want to allow distance greater than thicknessAtTrench in the kDTree query, 
#as some of these distances will not be orthogonal to the marker line, the dot product in the function will project these distances onto the normal vector
#We'll cull distances greater than thicknessAtTrench with a numpy boolean slice - this helps things work parallel


# In[84]:

slabXConds = operator.and_(xFn > slabxs.min(), xFn < slabxs.max())
slabYConds = depthFn < 1. - slabys.min()

#Two functions we'll use to limit the region of the initial thermal stancil
slabRegion =  fn.branching.conditional([(operator.and_(slabXConds,slabYConds), True),
                          (True, False)])

#slabCirc = inCircleFnGenerator((ndp.subZoneLoc, 1.0 - ndp.maxDepth), ndp.maxDepth)


# In[85]:

bufferlength = 1e3/sf.lengthScale

plateDepthFn = fn.branching.conditional([(depthFn < thicknessAtTrench, depthFn),
                                        (True, 1.)])

plateTempProxFn = ndp.potentialTemp*fn.math.erf((plateDepthFn)/(2.*fn.math.sqrt(1.*proxyageFn)))

slabTempProx  = ndp.potentialTemp*fn.math.erf((signedDistanceVariable)/(2.*np.sqrt(1.*ndp.slabMaxAge)))


proxytempConds = fn.branching.conditional([(signedDistanceVariable < bufferlength, plateTempProxFn),
                          #(operator.and_(slabRegion, slabCirc), fn.misc.min(slabTempProx , plateTempProxFn)),
                          (slabRegion,  fn.misc.min(slabTempProx , plateTempProxFn)),                 

                          (True, plateTempProxFn)]) 


#*************CHECKPOINT-BLOCK**************#

if not cp.restart:
    proxyTempVariable.data[:] = proxytempConds.evaluate(swarm)

#*************CHECKPOINT-BLOCK**************#


# ### marker
# 
# In this notebook, the markerLine helps us set up the initial distribution of weak material. We take the points that desribe the top of the slab, and we add extra particles along the top of the lower plate (so that there is weak material covering the entirety of the lower plate and slab)

# In[86]:

morexs = np.arange(mesh.minCoord[0] + 100e3/sf.lengthScale, ndp.subZoneLoc, ds)[:-1]
moreys = mesh.maxCoord[1]*np.ones(morexs.shape)


# In[87]:

#Build marker: copy the slab line, then move using the normal vector (director)

markerxs = np.concatenate((morexs,slabxs[:-2]))
markerys = np.concatenate((moreys,slabys[:-2]))
marker = markerLine2D(mesh, velocityField,markerxs, markerys, ndp.crustThickness,  1)

with marker.swarm.deform_swarm():
    marker.swarm.particleCoordinates.data[:] += marker.director.data*ndp.crustThickness
    

marker.rebuild()
marker.swarm.update_particle_owners()


# In[88]:

#inform the mesh of the marker

sd, pts0 = marker.compute_signed_distance(swarm.particleCoordinates.data, distance=thicknessAtTrench)
sp, pts0 = marker.compute_marker_proximity(swarm.particleCoordinates.data, distance=ndp.crustThickness)

#*************CHECKPOINT-BLOCK**************#
if not cp.restart:
    materialVariable.data[np.logical_and(sd<0,sp == marker.ID)] = sp[np.logical_and(sd<0,sp == marker.ID)]
#*************CHECKPOINT-BLOCK**************#



# ## Interpolate to temperature field
# 

# In[89]:

def swarmToTemp():

    _ix, _weights, _dist = nn_evaluation(swarm.particleCoordinates.data, mesh.data, n=4, weighted=True)


    #_dist.shape, mesh.data.shape
    #if 
    tempMapTol = 0.2
    tempMapMask = _dist.min(axis=1) < tempMapTol*(1. - mesh.minCoord[1])/mesh.elementRes[1] 
    
    #temperatureField.data[:] = 0.
    temperatureField.data[:] = ndp.potentialTemp #first set to dimensionless potential temp

    #now used IDW to assign temp from particles to Field
    #this is looking pretty ugly; nn_evaluation could use some grooming
    temperatureField.data[:,0][tempMapMask] = np.average(proxyTempVariable.evaluate(swarm)[_ix][tempMapMask][:,:,0],weights=_weights[tempMapMask], axis=1)

    #now cleanup any values that have fallen outside the Bcs

    temperatureField.data[temperatureField.data > 1.] = ndp.potentialTemp
    temperatureField.data[temperatureField.data < 0.] = 0.
    
    #and cleanup the BCs
    
    temperatureField.data[bWalls.data] = ndp.potentialTemp
    temperatureField.data[tWalls.data] = 0.


# In[90]:

#map proxy temp (swarm var) to mesh variable

if not cp.restart:
    swarmToTemp()


# ## choose temp field to use

# In[91]:

if md.thermal:
    temperatureFn = temperatureField
else:
    temperatureFn = proxyTempVariable


# ## adiabatic temp correction

# In[92]:

#Adiabatic correction: this is added to the arrhenius laws to simulate the adiabatic component
adiabaticCorrectFn =  ndp.potentialTemp*fn.math.exp(ndp.dissipation*depthFn) - ndp.potentialTemp



# In[93]:

#figTemp= glucifer.Figure(quality=3, boundingBox= bBox)
#figTemp.append( glucifer.objects.Points(swarm, temperatureFn + adiabaticCorrectFn, pointSize=1))
#figTempappend( glucifer.objects.Points(swarm, materialVariable, pointSize=1))
#figTemp.show()


# ## Rheology

# In[94]:


symStrainrate = fn.tensor.symmetric( 
                            velocityField.fn_gradient )

#Set up any functions required by the rheology
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))



def safe_visc(func, viscmin=ndp.viscosityMin, viscmax=ndp.viscosityMax):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))



# In[95]:

##Diffusion Creep
diffusionUM = (1./ndp.diffusionPreExp)*            fn.math.exp( ((ndp.diffusionEnergy + (depthFn*ndp.diffusionVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionLM = ndp.lowerMantleViscFac*(1./ndp.lowerMantlePreExp)*            fn.math.exp( ((ndp.lowerMantleEnergy + (depthFn*ndp.lowerMantleVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))


    
    
diffusion = fn.branching.conditional( ((depthFn < ndp.lowerMantleDepth, diffusionUM ), 
                                           (True,                      diffusionLM )  ))



    
diffusion = safe_visc(diffusion, viscmax=1e5)


    
#mantle Plasticity
ys =  ndp.cohesionMantle + (depthFn*ndp.frictionMantle)
ysf = fn.misc.min(ys, ndp.yieldStressMax)
yielding = ysf/(2.*(strainRate_2ndInvariant) + 1e-15) 

##Crust plasticity
crustys =  ndp.cohesionCrust + (depthFn*ndp.frictionCrust)
crustysf = fn.misc.min(crustys, ndp.yieldStressMax)
crustyielding0 = crustysf/(2.*(strainRate_2ndInvariant) + 1e-15) 


#This bit phases out the weak crust (effective plastic viscosity and visc. max) over specified depths
depthTaperFn = cosine_taper(depthFn, ndp.crustViscCutoffDepth, ndp.crustViscEndWidth)
crustyielding = crustyielding0*(1. - depthTaperFn) + depthTaperFn*yielding
viscmaxCrustFn = ndp.viscosityMaxCrust*(1. - depthTaperFn) + depthTaperFn*ndp.viscosityMax


#combined rheologies for mantle and weak layer (crust / interface)
mantleViscosityFn = safe_visc(fn.misc.min(diffusion, yielding), viscmin=ndp.viscosityMin, viscmax=ndp.viscosityMax)
crustViscosityFn = safe_visc(crustyielding, viscmin=ndp.viscosityMinCrust, viscmax=viscmaxCrustFn)


# In[96]:

viscosityMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {0:mantleViscosityFn,
                                    1:crustViscosityFn} )


# ## Buoyancy

# In[97]:

#Thermal Buoyancy

z_hat = ( 0.0, -1.0 )


thermalBuoyancyFn = ndp.rayleigh*(1. - temperatureFn)
thermalBuoyancyFn *=z_hat


# ## Any other functions we'll need

# In[98]:

###################
#Create integral, max/min templates 
###################

globRestFn = 1.

def surfint(Fn = 1., rFn=globRestFn, surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"]):
    return uw.utils.Integral( Fn*rFn, mesh=mesh, integrationType='Surface', surfaceIndexSet=surfaceIndexSet)



# ## Stokes system and solver

# In[99]:

print('got to Stokes')


# In[100]:

stokesPIC = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [freeslipBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = thermalBuoyancyFn )



# In[101]:

solver = uw.systems.Solver(stokesPIC)


if md.penaltyMethod:
    solver.set_inner_method("mumps")
    solver.options.scr.ksp_type="cg"
    solver.set_penalty(1.0e7)
    solver.options.scr.ksp_rtol = 1.0e-4

else:
    solver.options.main.Q22_pc_type='gkgdiag'
    solver.options.scr.ksp_rtol=5e-5
    solver.set_inner_method('mg')
    solver.options.mg.levels = 4
    
    
    
#avoid this initial solve if restarting
if not cp.restart:
    solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol)
    solver.print_stats()


# In[102]:

#remove drift in pressure
_pressure = surfint(pressureField)
_surfLength = surfint()
surfLength = _surfLength.evaluate()[0]

pressureSurf = _pressure.evaluate()[0]   
pressureField.data[:] -= pressureSurf/surfLength


# In[103]:

#figVel= glucifer.Figure(quality=3, boundingBox= bBox)
#figVel.append( glucifer.objects.Points(swarm, fn.math.dot(velocityField,velocityField), pointSize=1))
#figVel.append( glucifer.objects.VectorArrows(mesh, velocityField, arrowHead=5, scaling=0.0005))
#figVel.show()


# ## Setup advection-diffusion, swarm advection

# In[104]:

advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )

population_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.006, splitThreshold=0.25, maxDeletions=1, maxSplits=3, aggressive=True,aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))



if md.thermal:
    advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField,
                                         fn_sourceTerm    = 0.0,
                                         fn_diffusivity = 1., 
                                         #conditions     = [neumannTempBC, dirichTempBC] )
                                         conditions     = [ dirichTempBC] )


# ## Viz

# In[105]:

#Build a depth dependent mask for the vizualisation
#This helps reduce the size of glucifer databases

depthVariable      = swarm.add_variable( dataType="float", count=1 )
depthVariable.data[:] = depthFn.evaluate(swarm)

vizVariable      = swarm.add_variable( dataType="int", count=1 )
vizVariable.data[:] = 0

for index, value in enumerate(depthVariable.data[:]):
    #print index, value
    if np.random.rand(1)**30 > value/(mesh.maxCoord[1] - mesh.minCoord[1]):
        vizVariable.data[index] = 1

        
        
del index, value    #get rid of any variables that might be pointing at the .data handles (these are!)

#Now randomly cull more particles 
removeRandom = True
if removeRandom:
    reducFac = 0.6  #0.9 > remove 90%
    nonzs = np.where(vizVariable.data[:,0] == 1)[0].copy()
    nstart =  nonzs.shape[0]
    nend = int(np.ceil(nstart*reducFac))
    np.random.shuffle(nonzs)
    nstart, nend

    vizVariable.data[nonzs[:nend]] = 0
    del nonzs, nstart, nend


# In[106]:

#Set up the gLucifer stores

fullpath = os.path.join(outputPath + "gldbs/")
store1 = glucifer.Store(fullpath + 'subduction1.gldb')
store2 = glucifer.Store(fullpath + 'subduction2.gldb')



fig1 = glucifer.Figure(store1,figsize=(300*np.round(md.aspectRatio,2),300))
if md.thermal:
    fig1.append( glucifer.objects.Points(swarm, temperatureField, pointSize=2,  valueRange=[0.0, 1.0], fn_mask=vizVariable))
else:
    fig1.append( glucifer.objects.Points(swarm, proxyTempVariable, pointSize=2, valueRange=[0.0, 1.0],  fn_mask=vizVariable))

 


fig2 = glucifer.Figure(store2,figsize=(300*np.round(md.aspectRatio,2),300))
fig2.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2, fn_mask=vizVariable, logScale=True, valueRange=[100.*ndp.viscosityMin, ndp.viscosityMax]))
fig2.append( glucifer.objects.VectorArrows(mesh, velocityField, arrowHead=1, scaling=0.00005))




# ## Update functions for main loop

# In[107]:

def main_update(next_image_step):
    
    """
    This includes some functionality for image / file writing at a specified frequency,
    Assumes global variables:
        time, step, files_freq, next_image_step
    
    if numerical dt exceeds next specified writing point
    override dt make sure we hit that point
    Set some flags so that image / file writing proceeds
    
    """
    
    
    if md.thermal:
        dt = advDiff.get_max_dt()*md.courantFac #additional md.courantFac helps stabilise advDiff
        advDiff.integrate(dt)
        
    else:
        dt = advector.get_max_dt()
        
    #This relates to file writing at set period:
    #override dt make sure we hit certain time values
    #Set some flags so that image / file writing proceeds
    
    if step == 0:
        files_this_step = True
    else:
        files_this_step = False
    
    if time + dt >= next_image_step:
        dt = next_image_step - time
        files_this_step = True
        next_image_step += files_freq #increment time for our next image / file dump
        
        
    #Do advection    
        
    advector.integrate(dt)
    marker.advection(dt)
    
    #remove drift in pressure
    pressureSurf = _pressure.evaluate()[0]   
    pressureField.data[:] -= pressureSurf/surfLength
    
    
    return time+dt, step+1, files_this_step, next_image_step
    


# In[108]:

def viz_update():
    
    #Rebuild the viz. mask
    vizVariable.data[:] = 0

    for index, value in enumerate(depthVariable.data[:]):
        #print index, value
        if np.random.rand(1)**5 > value/(mesh.maxCoord[1] - mesh.minCoord[1]):
            vizVariable.data[index] = 1

    del index, value    #get rid of any variables that might be pointing at the .data handles
    
    if removeRandom:
        reducFac = 0.6  #0.9 > remove 90%
        nonzs = np.where(vizVariable.data[:,0] == 1)[0].copy()
        nstart =  nonzs.shape[0]
        nend = int(np.ceil(nstart*reducFac))
        np.random.shuffle(nonzs)
        nstart, nend

        vizVariable.data[nonzs[:nend]] = 0
        del nonzs, nstart, nend   
    
    #save gldbs
    fullpath = os.path.join(outputPath + "gldbs/")
    
    store1.step = step
    fig1.save( fullpath + "Temp" + str(step).zfill(5))
    
    store2.step = step
    fig2.save( fullpath + "visc" + str(step).zfill(5))

    


# In[109]:

def swarm_update():
    
    #run swarm repopulation
    population_control.repopulate()
    
    
    


# In[110]:

def markerLine_update():
    
    #cull particles once they reach a given depth
    
    cutoffDepth = 250e3/sf.lengthScale
    mask = marker.swarm.particleCoordinates.data[:,1] < (mesh.maxCoord[1] - cutoffDepth)

    with marker.swarm.deform_swarm():
        marker.swarm.particleCoordinates.data[mask] = (999999.,999999.)
        


# In[111]:

def xdmfs_update():
    
    #define any NN interps we'll need
    ix1, weights1, d1 = nn_evaluation(swarm.particleCoordinates.data, mesh.data, n=5, weighted=True)
    
    
    #rebuild any mesh vars that are not self-updating
    viscosityField.data[:,0] =  np.average(viscosityMapFn.evaluate(swarm)[:,0][ix1], weights=weights1, axis=len((weights1.shape)) - 1)
    strainRateField.data[:] = strainRate_2ndInvariant.evaluate(mesh)
     
    
    fullpath = os.path.join(outputPath + "xdmf/")
    #if not os.path.exists(fullpath+"mesh.h5"):
    #    _mH = mesh.save(fullpath+"mesh.h5")
    
    try:
        _mH
    except:
        _mH = mesh.save(fullpath+"mesh.h5")
    
    
    #Part 1
    mh = _mH
    vH = velocityField.save(fullpath + "velocity_" + str(step) +".h5")
    tH = temperatureField.save(fullpath + "temp_" + str(step) + ".h5")
    srH = strainRateField.save(fullpath + "strainrate_" + str(step) +".h5")
    viscH = viscosityField.save(fullpath + "visc_" + str(step) + ".h5")
    
    #part2
    
    velocityField.xdmf(fullpath + "velocity_" + str(step), vH, 'velocity', mh, 'mesh', modeltime=time)
    temperatureField.xdmf(fullpath + "temp_" + str(step), tH, 'temperature', mh, 'mesh', modeltime=time)
    strainRateField.xdmf(fullpath + "strainrate_" + str(step), srH, 'strainrate', mh, 'mesh', modeltime=time)
    viscosityField.xdmf(fullpath + "visc_" + str(step), viscH, 'visc', mh, 'mesh', modeltime=time)


# ## Main loop

# In[ ]:




# In[112]:

time = cp.time()  # Initial time
step = cp.step()   # Initial timestep
steps_output = 5   # output every 10 timesteps
metrics_output = 5
files_output = 10

files_freq  = md.filesFreqYears*(3600.*365.*24.)/sf.time  #applies to files and gldbs
files_this_step = False
next_image_step = (np.floor(time/files_freq)+ 1.) *files_freq 


# In[ ]:

#checkpoint at time zero
if not cp.restart:
    cp.saveObjs(step, time)
    cp.saveDicts(step, time)


# In[ ]:

while step < md.maxSteps:
    
    solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol)
    
    # main
    time,step,files_this_step, next_image_step = main_update(next_image_step)
    
    #markers / markerLines
    if step % md.swarmUpdate == 0:
        markerLine_update()    
    
    #checkpoint
    if step % md.checkpointEvery == 0:
        cp.saveObjs(step, time)
        cp.saveDicts(step, time)
        
    #particles
    if step % md.swarmUpdate == 0:
        swarm_update()
        
        
    #Viz
    if files_this_step:
        viz_update()    
        
    #xdmfs
    if files_this_step:
        xdmfs_update()
    
        
    print 'step =',step
    

    

print 'step =',step


# In[ ]:

#Done!

