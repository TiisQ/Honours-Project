import numpy as np

def getSParams(ArrayFile,FieldFile): ## must be strings
	array= {}
	field= {}
	with open(ArrayFile,'r') as ainfo:
		for line in ainfo:
			key, value= line.split(':')
			array[key]= value.strip()
        
	with open(FieldFile,'r') as finfo:
		for line in finfo:
			key, value= line.split(':')
			field[key]= value.strip()
	#I might have to further format this to allow for multiple sets of array/field profile information.		
	return array,field

def formatSParams(ArrayFile,FieldFile,td): ## isn't there a way of declaring this function without arguments?
	array,field= getSParams(ArrayFile,FieldFile)

	freq = np.float64(array['Centre_Freq'])
	ArNum = np.uint64(array['Num_Array'])
	## I may need a conditional/exception statement here incase the Num_Array does not correspond with the Centre_Dist info in file

	arrayHxpos= [np.float64(x) for x in array['Centre_Dist'].split(',')]
	arrayHxpos.sort()  ## sorting list in ascending order

	[Hmin,Hmax]= [np.float64(h) for h in field['HourAngleInterval'].split(',')]
	declangle= np.float64(field['FieldCentreDecl'])
	SNum= np.uint64(field['NumSources'])
	## I need a conditional statement here incase the NumSources does not correspond with the SourceParams info in file
	#using floats for the sake of generality. 
	
	point_sources=[]
	Amp=[]
	l=[]
	m=[]
	for sp in field['SourceParams'].split(' '):
		point_sources.append([np.float64(x) for x in sp.split(',')])
		
	for i in range(SNum):
		Amp.append(point_sources[i][0])
		l.append(point_sources[i][1]) ## in degrees, must be changed to radians in script
		m.append(point_sources[i][2])
		
	## baseline as a func of wavelength
	## I need to save both b12 and b21 for completion of ellipse
	lamda= 3e8/freq #speed of light c=3e8 m/s; lamda is now in units of m
	bline= [[(arrayHxpos[i]-arrayHxpos[k])/lamda for i in range(ArNum)]for k in range(ArNum)]

	#setting up my (point source) field  
	Hangles = np.linspace(Hmin,Hmax,td) # 12 hours because we are saving info about b12 AND b21
	Ht= 15*np.radians(Hangles)  #Hour angle H(t) has to be in radians which is *pi/12h and since 360deg=24hrs
	declrad= np.radians(declangle)

	## determining uv co-ordinates for uv tracks
	ut= np.array([[bline[k][i]*np.cos(Ht) for i in range(ArNum)]for k in range(ArNum)],dtype=np.float64)
	vt= np.array([[bline[k][i]*np.sin(Ht)*np.sin(declrad) for i in range(ArNum)]for k in range(ArNum)],dtype=np.float64)
		
	return Amp,l,m,ut,vt,arrayHxpos
  
## maybe I could adapt this Amp,l,m to be the model parameters instead of them being vestigial
