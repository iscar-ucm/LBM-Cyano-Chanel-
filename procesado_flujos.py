# En este archivo se procesan los datos obtenidos de una simulación LBM
#para que se puedan usar directamente en el modelo de automata celular de CELL-DEVS.
# Para ello hay que marcar las líneas donde estarían las interfaces de las celdas
#e integrar el flujo normal a cada una. Finalmente guardarlo de una manera ordenada para que
#CELL-DEVS pueda leerlo de manera sistemática.

import numpy as np
import pandas as pd
import os


warmup = 220
datafiles = os.listdir("Output/Data")
data = pd.read_csv("Output/Data/Out_000.csv") #[x,y,z,u,v,w,d]

# Dimensiones de la malla
n_x,n_y,n_z = np.max(data[['x','y','z']].values,axis=0) + 1
obs_x = n_x / 4
obs_y = n_y / 2
obs_r = n_y / 6

notobstacle = np.sqrt((data['x']-obs_x)**2 + (data['y']-obs_y)**2) > obs_r
dlimobjx = obs_x - np.sqrt(obs_x**2-(-obs_r**2+(obs_y-20)**2+obs_x**2))
ulimobjx = obs_x + np.sqrt(obs_x**2-(-obs_r**2+(obs_y-25)**2+obs_x**2))
dlimobjy = obs_y - np.sqrt(obs_y**2-(-obs_r**2+(obs_x-35)**2+obs_y**2))
ulimobjy = obs_y + np.sqrt(obs_y**2-(-obs_r**2+(obs_x-35)**2+obs_y**2))

#Vertices que definen las regiones del escenario 2D
#Se sigue un orden de izquierda a derecha y de abajo a arriba
vert = [[0,0],[50,0],
        [18,0],[18,20],[18,30],[18,50],
        [35,0],[35,10],[35,20],[35,30],[35,40],[35,40],
        [50,10],[50,25],[50,40],
        [60,0],[60,10],[60,25],[60,40],[60,50],
        [85,0],[85,10],[85,25],[85,40],[85,50],
        [105,0],[105,10],[105,25],[105,40],[105,50],
        [125,0],[125,10],[125,25],[125,40],[125,50],
        [145,0],[145,10],[145,25],[145,40],[145,50],
        [170,0],[170,10],[170,25],[170,40],[170,50]]

vert_z = np.linspace(0,n_z,11).astype(int)
dz = vert_z[1]-vert_z[0]

f={} #keys definidas como ((x1,y1,z1),(x2,y2,z2)) para las fronteras.
#(x1,y1,z1) siempre será con x1<x2, si =: y1<y2,...
inic = 1
for file in datafiles[warmup:800]:
    data = pd.read_csv('Output/Data/'+file) #[t,x,y,z,u,v,w,d]
    #Frontera C0-T1
    limx=18
    dlimy=30
    ulimy=50
    surf= (ulimy-dlimy) * dz #/50
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=dlimy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,0,k),(0,1,k))] = u_mean[0]*surf
        else:
            f[((0,0,k),(0,1,k))] = np.append(f[((0,0,k),(0,1,k))],u_mean[0]*surf)

    #Frontera C0-C1
    limx=18
    ulimy=30
    dlimy=20
    surf= (ulimy-dlimy) *dz #/50
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values<ulimy,data["y"].values>=dlimy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,0,k),(0,2,k))] = u_mean[0]*surf
        else:
            f[((0,0,k),(0,2,k))] = np.append(f[((0,0,k),(0,2,k))],u_mean[0]*surf)

    #Frontera C0-T4
    limx=18
    ulimy=20
    surf=20*dz#/50
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values<ulimy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,0,k),(0,3,k))] = u_mean[0]*surf
        else:
            f[((0,0,k),(0,3,k))] = np.append(f[((0,0,k),(0,3,k))],u_mean[0]*surf)

    #Frontera T1-T2
    ulimx=35
    dlimx=18
    ulimy=40
    dlimy=30
    m=(ulimy-dlimy)/(ulimx-dlimx)
    lup=np.sqrt((ulimx-dlimx)**2+(ulimy-dlimy)**2)
    ppvert = [1,-1/m]/np.sqrt(1+1/m**2)
    surf=lup*dz#/(lup+dlimy)
    for k in range(len(vert_z)-1):
        ind=np.vstack([np.arange(dlimx,ulimx),np.round(dlimy+m*(np.arange(dlimx,ulimx)-dlimx))]).T
        aux = np.empty((0,2))
        for l in range(len(ind)):
            dataxy = data[["u","v"]][np.all([data["x"].values==ind[l,0],data["y"].values==ind[l,1],
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
            aux = np.vstack([aux,dataxy.values])
        u_mean = np.nanmean(aux,axis=0)
        if inic:
            f[((0,1,k),(1,0,k))] = np.dot(u_mean,ppvert)*surf
        else:
            f[((0,1,k),(1,0,k))] = np.append(f[((0,1,k),(1,0,k))],np.dot(u_mean,ppvert)*surf)

    #Frontera C1-T2
    ulimx=35
    dlimx=18
    limy=30
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<dlimobjx,data["y"].values==limy,
                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        surf=(dlimobjx-dlimx)*dz
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,2,k),(1,0,k))] = u_mean[1]*surf
        else:
            f[((0,2,k),(1,0,k))] = np.append(f[((0,2,k),(1,0,k))],u_mean[1]*surf)

    #Frontera C1-T3
    ulimx=35
    dlimx=18
    limy=20
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<dlimobjx,data["y"].values==limy,
                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        surf=(dlimobjx-dlimx)*dz
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,2,k),(1,3,k))] = u_mean[1]*surf
        else:
            f[((0,2,k),(1,3,k))] = np.append(f[((0,2,k),(1,3,k))],u_mean[1]*surf)

    #Frontera T4-T3
    ulimx=35
    dlimx=18
    ulimy=20
    dlimy=10
    surf=lup*dz#/(lup+30)
    m=(ulimy-dlimy)/(ulimx-dlimx)
    ppvert = [1,1/m]/np.sqrt(1+1/m**2)
    for k in range(len(vert_z)-1):
        ind=np.vstack([np.arange(dlimx,ulimx),np.round(ulimy-m*(np.arange(dlimx,ulimx)-dlimx))]).T
        aux = np.empty((0,2))
        for l in range(len(ind)):
            dataxy = data[["u","v"]][np.all([data["x"].values==ind[l,0],data["y"].values==ind[l,1],
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
            aux = np.vstack([aux,dataxy.values])
        u_mean = np.nanmean(aux,axis=0)
        if inic:
            f[((0,3,k),(1,3,k))] = np.dot(u_mean,ppvert)*surf
        else:
            f[((0,3,k),(1,3,k))] = np.append(f[((0,3,k),(1,3,k))],np.dot(u_mean,ppvert)*surf)

    #Frontera T1-C2
    limx=35
    dlimy=40
    surf=10*dz#/(lup+30)
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=dlimy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,1,k),(2,0,k))] = u_mean[0]*surf
        else:
            f[((0,1,k),(2,0,k))] = np.append(f[((0,1,k),(2,0,k))],u_mean[0]*surf)

    #Frontera T2-T5
    limx=35
    ulimy=40
    dlimy=30
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=ulimobjy,data["y"].values<ulimy,
                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        surf=(40-ulimobjy)*dz
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,0,k),(1,1,k))] = u_mean[0]*surf
        else:
            f[((1,0,k),(1,1,k))] = np.append(f[((1,0,k),(1,1,k))],u_mean[0]*surf)

    #Frontera C2-T5
    ulimx=50
    dlimx=35
    limy=40
    surf=(ulimx-dlimx)*dz#/(20+60-dlimx)
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values<ulimx,data["x"].values>=dlimx,data["y"].values==limy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,1,k),(2,0,k))] = u_mean[1]*surf
        else:
            f[((1,1,k),(2,0,k))] = np.append(f[((1,1,k),(2,0,k))],u_mean[1]*surf)

    #Frontera T5-T6
    ulimx=50
    dlimx=35
    limy=25
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=ulimobjx,data["x"].values<ulimx,data["y"].values==limy,
                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        surf=(ulimx-ulimobjx)*dz
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,1,k),(1,2,k))] = u_mean[1]*surf
        else:
            f[((1,1,k),(1,2,k))] = np.append(f[((1,1,k),(1,2,k))],u_mean[1]*surf)

    #Frontera T3-T6
    limx=35
    ulimy=20
    dlimy=10
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=dlimy,data["y"].values<dlimobjy,
                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        surf=(dlimobjy-dlimy)*dz
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,2,k),(1,3,k))] = u_mean[0]*surf
        else:
            f[((1,2,k),(1,3,k))] = np.append(f[((1,2,k),(1,3,k))],u_mean[0]*surf)

    #Frontera T4-C3
    limx=35
    ulimy=10
    surf=ulimy*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values<ulimy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((0,3,k),(2,3,k))] = u_mean[0]*surf
        else:
            f[((0,3,k),(2,3,k))] = np.append(f[((0,3,k),(2,3,k))],u_mean[0]*surf)

    #Frontera T6-C3
    ulimx=50
    dlimx=35
    limy=10
    surf=(ulimx-dlimx)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values==limy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,2,k),(2,3,k))] = u_mean[1]*surf
        else:
            f[((1,2,k),(2,3,k))] = np.append(f[((1,2,k),(2,3,k))],u_mean[1]*surf)

    #Frontera T5-C4
    limx=50
    dlimy=25
    ulimy=40
    surf=(ulimy-dlimy)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=dlimy,data["y"].values<ulimy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)*surf
        if inic:
            f[((1,1,k),(2,1,k))] = u_mean[0]
        else:
            f[((1,1,k),(2,1,k))] = np.append(f[((1,1,k),(2,1,k))],u_mean[0])

    #Frontera C2-C4
    dlimx=50
    ulimx=65
    limy=40
    surf=(ulimx-dlimx)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values==limy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((2,0,k),(2,1,k))] = u_mean[1]*surf
        else:
            f[((2,0,k),(2,1,k))] = np.append(f[((2,0,k),(2,1,k))],u_mean[1]*surf)

    #Frontera C4-C5
    ulimx=65
    dlimx=50
    limy=25
    surf=(ulimx-dlimx)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values==limy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((2,1,k),(2,2,k))] = u_mean[1]*surf
        else:
            f[((2,1,k),(2,2,k))] = np.append(f[((2,1,k),(2,2,k))],u_mean[1]*surf)

    #Frontera T6-C5
    limx=50
    dlimy=10
    ulimy=25
    surf=(ulimy-dlimy)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values==limx,data["y"].values>=dlimy,data["y"].values<ulimy,
                                        data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((1,2,k),(2,2,k))] = u_mean[0]*surf
        else:
            f[((1,2,k),(2,2,k))] = np.append(f[((1,2,k),(2,2,k))],u_mean[0]*surf)

    #Frontera C5-C3
    dlimx=50
    ulimx=65
    limy=10
    surf=(ulimx-dlimx)*dz
    for k in range(len(vert_z)-1):
        dataxy=data[["u","v"]][np.all([data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values==limy,
                                            data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
        u_mean = np.nanmean(dataxy,axis=0)
        if inic:
            f[((2,2,k),(2,3,k))] = u_mean[1]*surf
        else:
            f[((2,2,k),(2,3,k))] = np.append(f[((2,2,k),(2,3,k))],u_mean[1]*surf)

    #Fronteras mallas 65+. Se guardan las relaciones de la pared izquierda e superior (XY), y superior (XZ) de cada celda.
    x_ticks=np.arange(65,n_x,20)
    x_ticks=np.append(x_ticks,n_x-1)
    y_ticks=np.array([0,10,25,40,50])
    dataxy=np.zeros([len(x_ticks),len(y_ticks),3],dtype=object)
    u_mean = np.zeros([len(x_ticks),len(y_ticks),3])
    i0=3
    jmax= len(y_ticks)-2
    for k in range(len(vert_z)-1):
        for i in range(len(x_ticks)-1):
            for j in range(len(y_ticks)-1):
                surfx=(x_ticks[i+1]-x_ticks[i])*dz
                surfy=(y_ticks[j+1]-y_ticks[j])*dz
                surfz=(y_ticks[j+1]-y_ticks[j])*(x_ticks[i+1]-x_ticks[i])
                dataxy[i,j,0]=data[["u"]][np.all([data["y"].values>=y_ticks[j],data["y"].values<y_ticks[j+1],
                                                data["x"].values==x_ticks[i],data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
                u_mean[i,j,0]=np.nanmean(dataxy[i,j,0].values)*surfy
                if inic:
                    f[((i0+i-1,jmax-j,k),(i0+i,jmax-j,k))]=u_mean[i,j,0]
                else:
                    f[((i0+i-1,jmax-j,k),(i0+i,jmax-j,k))]=np.append(f[((i0+i-1,jmax-j,k),(i0+i,jmax-j,k))],u_mean[i,j,0])

                if j<len(y_ticks)-2:
                    dataxy[i,j,1]=data[["v"]][np.all([data["x"].values>=x_ticks[i],data["x"].values<x_ticks[i+1],
                                                    data["y"].values==y_ticks[j+1],data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
                    u_mean[i,j,1]=np.nanmean(dataxy[i,j,1].values)*surfx
                    if inic:
                        f[((i0+i,jmax-j,k),(i0+i,jmax-(j+1),k))]=u_mean[i,j,1]
                    else:
                        f[((i0+i,jmax-j,k),(i0+i,jmax-(j+1),k))]=np.append(f[((i0+i,jmax-j,k),(i0+i,jmax-(j+1),k))],u_mean[i,j,1])

                if k<len(vert_z)-2:
                    dataxy[i,j,2]=data[["w"]][np.all([data["x"].values>=x_ticks[i],data["x"].values<x_ticks[i+1],
                                                    data["y"].values>=y_ticks[j],data["y"].values<y_ticks[j+1],data['z'].values==vert_z[k+1]],axis=0)]
                    u_mean[i,j,2]=np.nanmean(dataxy[i,j,2].values)*surfz
                    if inic:
                        f[((i0+i,jmax-j,k),(i0+i,jmax-j,k+1))]=u_mean[i,j,2]
                    else:
                        f[((i0+i,jmax-j,k),(i0+i,jmax-j,k+1))]=np.append(f[((i0+i,jmax-j,k),(i0+i,jmax-j,k+1))],u_mean[i,j,2])
                
        for j in range(len(y_ticks)-1):  #Pared en YZ final, sección de salida del flujo (x=n_x)
            dataxy[-1,j,0]=data[["u"]][np.all([data["y"].values>=y_ticks[j],data["y"].values<y_ticks[j+1],
                                               data["x"].values==x_ticks[-1],data['z'].values>=vert_z[k],data['z'].values<vert_z[k+1]],axis=0)]
            u_mean[-1,j,0]=np.nanmean(dataxy[-1,j,0].values)*(y_ticks[j+1]-y_ticks[j])*dz
            if inic:
                f[((i0+len(x_ticks)-2,jmax-j,k),(i0+len(x_ticks)-1,jmax-j,k))]=u_mean[-1,j,0]
            else:
                f[((i0+len(x_ticks)-2,jmax-j,k),(i0+len(x_ticks)-1,jmax-j,k))]=np.append(f[((i0+len(x_ticks)-2,jmax-j,k),(i0+len(x_ticks)-1,jmax-j,k))],u_mean[-1,j,0])

    #Paredes horizontales XY rectangulares & compuestas
    dxdyC = {
        (0,0): [[0,0],[18,50]],
        (2,0): [[35,40],[65,50]],
        (2,3): [[35,0],[65,10]],
        (2,1): [[50,25],[65,40]],
        (2,2): [[50,10],[65,25]],
        (1,1): [[35,25],[50,40]],
        (1,2): [[35,10],[50,25]],
        (0,2): [[18,20],[35,30]]
    }
    corteCyp = ulimobjy
    corteCyn = dlimobjy
    corteCxn = dlimobjx; corteCxp=ulimobjx
    thetaT5 = np.pi-np.arcsin((corteCyp-obs_y)/obs_r)
    surfT5C = obs_r**2/2*(thetaT5 -np.sin(thetaT5)) + \
                (corteCyp-dxdyC[(1,1)][0][1])*(corteCxp-dxdyC[(1,1)][0][0])/2 #np.pi-arcsin because np.arcsin returns angles in [-pi/2,pi/2]
    thetaT6 = np.pi-np.arcsin((corteCyp-obs_y)/obs_r)
    surfT6C = obs_r**2/2*(thetaT6-np.sin(thetaT6)) + \
                (dxdyC[(1,2)][1][1]-corteCyn)*(corteCxp-dxdyC[(1,2)][0][0])/2
    theta = 2*np.arccos((obs_x-corteCxn)/obs_r)
    surfC1C = obs_r**2/2*(theta - np.sin(theta)) + (dxdyC[(0,2)][1][1]-dxdyC[(0,2)][0][1])*(dxdyC[(0,2)][1][0]-corteCxn)
    obs_effect = [0,0,0,0,0,surfT5C,surfT6C,surfC1C]
    for k in range(1,len(vert_z)-1):
        contador=0
        for key in dxdyC.keys():
            condition = np.all([notobstacle,data["x"].values>=dxdyC[key][0][0],data["x"].values<dxdyC[key][1][0],
                                            data["y"].values>=dxdyC[key][0][1],data["y"].values<dxdyC[key][1][1],
                                            data['z'].values==vert_z[k]],axis=0)
            dataxy=data[["w"]][condition]
            surf=(dxdyC[key][1][0]-dxdyC[key][0][0])*(dxdyC[key][1][1]-dxdyC[key][0][1])-obs_effect[contador]
            u_mean = np.nanmean(dataxy.values,axis=0)[0]
            if inic:
                f[(key+(k-1,),key+(k,))] = u_mean*surf
            else:
                f[(key+(k-1,),key+(k,))] = np.append(f[(key+(k-1,),key+(k,))],u_mean*surf)
            contador+=1

    #Paredes horizontales XY triangulares
    dxdyT = { #dlimy, ulimy --> 1D ó 2D]
        (0,1): [[30,40],50],
        (1,0): [30,[30,40]],
        (1,3): [[20,10],20],
        (0,3): [0,[20,10]]
    }
    ulimx= 35; dlimx= 18
    dx=ulimx-dlimx
    corteCxn = dlimobjx
    corteCyp = obs_y + np.sqrt(obs_y**2-(-obs_r**2+(obs_x-ulimx)**2+obs_y**2))
    corteCyn = obs_y - np.sqrt(obs_y**2-(-obs_r**2+(obs_x-ulimx)**2+obs_y**2))

    surfT2 = 0.5*dx*(40-30)
    alfa = np.arcsin((corteCyp-obs_y)/obs_r)-np.arcsin((dxdyT[(1,0)][0]-obs_y)/obs_r)
    surfT2C = surfT2 - (ulimx-corteCxn)*(corteCyp-dxdyT[(1,0)][0])/2 - obs_r**2/2*(alfa-np.sin(alfa))
    surfT1 = surfT2 + dx*(50-40)
    surfT3 = 0.5*dx*(20-10)
    beta = np.arcsin((dxdyT[(1,3)][1]-obs_y)/obs_r)-np.arcsin((corteCyn-obs_y)/obs_r)
    surfT3C = surfT3 - (ulimx-corteCxn)*(dxdyT[(1,3)][1]-corteCyn)/2 - obs_r**2/2*(beta-np.sin(beta))
    surfT4 = surfT3 + dx*(10-0)
    surf = [surfT1,surfT2C,surfT3C,surfT4]
    ids = [(0,1),(1,0),(1,3),(0,3)]

    for k in range(1,len(vert_z)-1):
        nsurf=0
        for j in ids:
            if np.size(dxdyT[j][0])==1:
                m=(dxdyT[j][1][1]-dxdyT[j][1][0])/dx
                dataxy=data[["w"]][np.all([notobstacle,data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values>=dxdyT[j][0],
                                            data["y"].values<dxdyT[j][1][0]+m*(data['x'].values-dlimx),data['z'].values==vert_z[k]],axis=0)]
                u_mean = np.nanmean(dataxy.values,axis=0)[0]
            else:
                m=(dxdyT[j][0][1]-dxdyT[j][0][0])/dx
                dataxy=data[["w"]][np.all([notobstacle,data["x"].values>=dlimx,data["x"].values<ulimx,data["y"].values>=dxdyT[j][0][0]+m*(data['x'].values-dlimx),
                                            data["y"].values<dxdyT[j][1],data['z'].values==vert_z[k]],axis=0)]
                u_mean = np.nanmean(dataxy.values,axis=0)[0]
            if inic:
                f[(j+(k-1,),j+(k,))] = u_mean*surf[nsurf]
            else:
                f[(j+(k-1,),j+(k,))] = np.append(f[(j+(k-1,),j+(k,))],u_mean*surf[nsurf])
            nsurf+=1
    inic=0
    print('%s done'%file)
dataout = pd.DataFrame(f,index=None)
dataout.to_pickle('Output/flowdata.pkl')
print('Proceso completado')