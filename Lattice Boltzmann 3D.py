import numpy as np 
from tqdm import tqdm
import os
import time
import pandas as pd
import shutil #Para eliminar directorios previos con rmtree

#Crear directorios
folder_padre = os.path.dirname(os.path.abspath(__name__))

folder_data = folder_padre + '/Output/Data'
if os.path.exists(folder_data):
    shutil.rmtree(folder_data)
os.makedirs(folder_data)

#Parámetros simulación
n_iter = 100000
plot_n_steps = 100
skip_first_iter = 0

#### Mundo físico  [X->ancho, Y->alto, Z->largo]
Lx = 150
Ly = 50
Lz = 50
nu = 4e-3
u = 0.05
Crho = 1000.0

#### Mundo Lattice
CL = 1 #Asegurarse de que de un número entero correcto al dividir Ls
n_x = round(Lx/CL)
n_y = round(Ly/CL)
n_z = round(Lz/CL)
    #Obstáculo
obs_x = n_x / 4
obs_y = n_y / 2
obs_z = n_z / 2
obs_r = n_y / 6

#Parámetros fluido
#Reynolds_number = 100  #Siempre igual en unidades físicas y lattice
Re = (u * obs_r*CL) / nu #Viscosidad cinemática [calculo dimensional] --> igual en lattice units
tau = 0.506 #3.0 * nu + 0.5 #Ajustar tau para alterar el paso de tiempo. A menor, mayor paso de tiempo --> CUIDADO CON LA ESTABILIDAD!!
omega = 1/tau
Ct = (tau-0.5)/3*CL**2/nu
Cu = CL/Ct
Cf = Crho*CL/Ct**2
gravity = -9.8 / Cf
inflow_vel = u / Cu
print(' Re = %.1f \t grav = %f \n Ct = %f \t Cu = %f \n inflow_v = %f'%(Re,gravity,Ct,Cu,inflow_vel))
if inflow_vel > 0.3:
    raise ValueError("La velocidad de entrada es muy alta, debe ser menor a 0.3. [u = %f]"%inflow_vel)

x = np.arange(n_x)
y = np.arange(n_y)
z = np.arange(n_z)
X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")


obstacle = np.sqrt((X-obs_x)**2 + (Y-obs_y)**2) < obs_r
obstacle[:, 0, :] = True
obstacle[:, -1, :] = True
obstacle[:, :, 0] = True
obstacle[:, :, -1] = True


#Velocidad
vel_profile = np.zeros((n_x, n_y, n_z, 3))
vel_profile[:, :, :, 0] = inflow_vel

n_discret_vel = 15

lattice_vel = np.array([
    [0, 1, 0, 0, -1,  0,  0, 1,  1,  1,  1, -1, -1, -1, -1],
    [0, 0, 1, 0,  0, -1,  0, 1,  1, -1, -1,  1,  1, -1, -1],
    [0, 0, 0, 1,  0,  0, -1, 1, -1,  1, -1,  1, -1,  1, -1]
])

lattice_ind = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
])

opposite_ind = np.array([
    0, 4, 5, 6, 1, 2, 3, 14, 13, 12, 11, 10, 9, 8, 7
])

lattice_w = np.array([
    2/9, 
    1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 
    1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72
])

x_neg_vel = np.array([4, 11, 12, 13, 14])
x_0_vel = np.array([0, 2, 3, 5, 6])
x_pos_vel = np.array([1, 7, 8, 9, 10])

y_neg_vel = np.array([5, 9, 10, 13, 14])
y_0_vel = np.array([0, 1, 3, 4, 6])
y_pos_vel = np.array([2, 7, 8, 11, 12])

z_neg_vel = np.array([6, 8, 10, 12, 14])
z_0_vel = np.array([0, 1, 2, 4, 5])
z_pos_vel = np.array([3, 7, 9, 11, 13])

#Gravedad
F = np.zeros((n_x, n_y, n_z, 3))
F[:, :, :, 2] = gravity

proj_force = np.einsum('LNMd,dQ->LNMQ', F, lattice_vel)
gravity_force = 3 * proj_force * (2 * tau - 1)/(2 * tau) 

#Funciones
def get_density(discrete_vel,axis=-1):
    density = np.sum(discrete_vel, axis=axis, dtype= np.float32)
    
    return density

def get_macro_vel(discrete_vel, density):
    macro_vel = np.einsum('LMNQ, dQ -> LMNd', discrete_vel, lattice_vel)/ density[..., np.newaxis]

    return macro_vel

def get_f_eq(macro_vel, density):
    gravity_macro_vel = macro_vel + (F / (2 * density[..., np.newaxis]))

    proj_discete_vel = np.einsum("dQ,LMNd->LMNQ", lattice_vel, gravity_macro_vel)
    
    macro_vel_mag = np.linalg.norm(gravity_macro_vel, axis=-1)
    
    f_eq = (density[..., np.newaxis] * lattice_w[np.newaxis, np.newaxis, np.newaxis, :] * (
            1 + 3 * proj_discete_vel + 9/2 * proj_discete_vel**2 - 3/2 * macro_vel_mag[..., np.newaxis]**2
        )
    )

    return f_eq

#----------------------- SIMULACIÓN -----------------------

def main():
    def update(discrete_vel_0):
        #(1) Frontera salida
        discrete_vel_0[-1, :, :, x_neg_vel] = discrete_vel_0[-2, :, :, x_neg_vel]

        #(2) Velocidades macro
        density_0 = get_density(discrete_vel_0)
        macro_vel_0 = get_macro_vel(discrete_vel_0, density_0)

        #(3) Frontera entrada Dirichlet
        macro_vel_0[0, :, :, :] = vel_profile[0, :, :, :]
        density_0[0, :, :] = (get_density(discrete_vel_0[0, :, :, x_0_vel],axis=0) + 2 * get_density(discrete_vel_0[0, :, :, x_neg_vel],axis=0)) / (1 - macro_vel_0[0, :, :, 0])
        #if np.any(density_0[0, :, :] <= 0):
        #    raise ValueError('Densidad negativa en la frontera de entrada')
        #(4) f_eq 
        f_eq = get_f_eq(macro_vel_0, density_0)

        #(3) 
        discrete_vel_0[0, :, :, x_pos_vel] = f_eq[0, :, :, x_pos_vel]

        #(5) Colisión BGK
        discrete_vel_1 = discrete_vel_0 - omega * (discrete_vel_0 - f_eq) + gravity_force

        #(6) Condiciones de frontera obstaculo
        for i in range(n_discret_vel):
            discrete_vel_1[obstacle, lattice_ind[i]] = discrete_vel_0[obstacle, opposite_ind[i]]

        #(7) Difusión
        discrete_vel_2 = discrete_vel_1
        for i in range(n_discret_vel):
            discrete_vel_2[:, :, :, i] = np.roll(
                np.roll(
                    np.roll(
                        discrete_vel_1[:, :, :, i],
                        lattice_vel[0, i],
                        axis=0,
                    ),
                    lattice_vel[1, i],
                    axis=1,
                ),
                lattice_vel[2, i],
                axis= 2,
            )

        return discrete_vel_2

    discrete_vel_0 = get_f_eq(vel_profile, np.ones((n_x, n_y, n_z)))

    n = 0 #Contador
    dat = 0 #Contador
    file_name_tiempos = os.path.join(folder_padre, 'Tiempos de simulación (LB3D).txt')

    for iter in tqdm(range(n_iter)):
        
        inicio = time.time()

        discrete_vel_1 = update(discrete_vel_0)
        discrete_vel_0 = discrete_vel_1

        final = time.time()
        tiempo_ejecucion = final - inicio

        
        #if n == 0:
        #    open(file_name_tiempos, 'w')
        open(file_name_tiempos, 'a').write('\n Frame %i, %f'%(n, tiempo_ejecucion))

        n += 1

        if iter % plot_n_steps == 0  and iter >= skip_first_iter:
            dat = iter // plot_n_steps

            density = get_density(discrete_vel_1)
            macro_vel = get_macro_vel(discrete_vel_1, density)*Cu
            data = pd.DataFrame({'t': iter*Ct, 'x':X.flatten(), 'y':Y.flatten(), 'z':Z.flatten(),
                                                 'u':macro_vel[:,:,:,0].flatten(), 'v':macro_vel[:,:,:,1].flatten(), 'w':macro_vel[:,:,:,2].flatten(),
                                                 'd':density.flatten()*Crho})
            frame_iter = "{:03d}".format(dat)
            file_data = os.path.join(folder_data, 'Out_%s.csv'%frame_iter)
            data.to_csv(file_data, index=False)
    
    timedata = pd.DataFrame({'maxtime': n_iter*Ct, 'dt': Ct },index = 0)
    timedata.to_csv(os.path.join(folder_data, 'TimeData.csv'), index=False)

if __name__ == '__main__':
    main()
