import argparse
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
import sys
import os

def distance(x0, x1, box):
    # xo is a position of one atom, x1 is an array of positions
    delta = np.abs(x0 - x1)
    delta = delta - box * np.round(delta/(box))
    return np.sqrt((delta ** 2).sum(axis=-1))

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

if __name__=="__main__":
    # main code
    parser = argparse.ArgumentParser(description='Program that computes positions of exclusion cylinders for CNM simulations subject to min dist criteria, sets initial atom locations and generates required files for holey-EDIP simulations.')
    #================#
    # ordered inputs #
    #================#
    #parser.add_argument('infile', metavar='infile', type=str, help='input xyz file.')
    #================#
    # optional flags #
    #================#
    parser.add_argument('--n',default=60, type=int, help='Nmuber of exclusion cylinders, Default= 60.')
    parser.add_argument('--natoms',default=7945, type=int, help='Number of atoms. Default= 7945.')
    parser.add_argument('--maxmoves',default=1000, type=int, help='Maximum number of cylinder move attempts. Default= 1000.')
    parser.add_argument('--maxcount_atom',default=10000, type=int, help='Maximum number of atom placement attempts. Default= 10000')
    parser.add_argument('--tol',default=1., type=float, help='Minimum distance between atoms. Default=1.')
    parser.add_argument('--sep',default=6., type=float, help='Minimum separation distance between cylinders, Default= 6.')
    parser.add_argument('--l',default=100., type=float, help='Box length in x and y. Default=100.')
    parser.add_argument('--z',default=12., type=float, help='Unit cell height. Default= 12.')
    parser.add_argument('--mean_poresize',default=7., type=float, help='Mean pore diameter for initial sample. Default= 7.')
    parser.add_argument('--std',default=1.25, type=float, help='Standard deviation of sampled pore diameter distribution. Default=1.25')
    parser.add_argument('--noplots', default=False, action='store_true',help='Flag that turns off post run plots.')
    parser.add_argument('--debug', default=False, action='store_true',help='Verbose flag.')
    parser.add_argument('--gen_slm', default=False, action='store_true',help='Generate slurm run script for magnus')
    parser.add_argument('--partition', default='workq', type=str, help='Name of simulation directory')
    parser.add_argument('--simtime', default=24, type=int, help='Simulation runtime in hours')
    parser.add_argument('--nnodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--outdir', default='heat_and_anneal', type=str, help='Name of simulation directory')

    # parse input and assign variables
    args = parser.parse_args()
    l = args.l
    z = args.z
    n = args.n
    natoms = args.natoms
    maxmoves = args.maxmoves
    maxcount_atom = args.maxcount_atom
    sep = args.sep
    tol = args.tol
    mean_poresize = args.mean_poresize
    std = args.std
    debug = args.debug
    noplots = args.noplots
    gen_slm = args.gen_slm
    partition = args.partition
    simtime = args.simtime
    nnodes = args.nnodes
    outdir = args.outdir

    # first check if outdirname already exists and add a number if so
    if os.path.exists(outdir + '.tar.gz'):
        outdir_base = outdir
        for i in range(1,101):
            outdir = outdir_base + f'_{i}'
            if not os.path.exists(outdir + '.tar.gz'):
                break
            elif os.path.exists(outdir + '.tar.gz') and i >=100:
                sys.exit()
    os.mkdir(outdir)
    print(f'Make directory {outdir}')

    # setup unit cell dimensions
    box = np.array([l, l, z])
    print(box)

    # setup random number generator
    rng = np.random.default_rng()
    # initialise cylinder positions
    xy = rng.random((n,2))*l
    r = 0.5*(rng.normal(mean_poresize, std, n))

    # if run in debug, graph iterative cylinder fitting
    if debug:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0,l)
        ax.set_ylim(0,l)
        ax.set_aspect('equal')
        plt.show()
        ax.paches = []
        color = plt.cm.gist_ncar(np.linspace(0, 1,n))
        for i in range(n):
            ax.add_patch(plt.Circle((xy[i]), radius=r[i], color = color[i]))

    # fitting cylinders, logic identical to holes2.m
    displace = 1
    moves = 0
    xy -= 0.5*l
    while (displace > 0) and (moves < maxmoves):
        moves += 1
        displace = 0
        for i in range(n-1):
            for j in range(i+1,n):
                delta = xy[i] - xy[j]
                delta -= box[:-1] * np.round(delta/box[:-1])
                rij = np.sqrt((delta**2).sum())
                gap = rij - r[i] - r[j]
                if (gap < sep):
                    noise =  rng.normal(0,0.5,2)
                    move = (sep-gap) * delta/rij
                    xy[i] += move
                    xy[j] -= move
                    displace += 1

        xy -= 0.5*l
        xy -= box[:-1] * np.round(xy/box[:-1])
        xy += 0.5*l

        if debug:
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)
            plt.pause(0.0001)
            ax.paches = []
            print(moves, displace)
            for i in range(n):
                ax.add_patch(plt.Circle((xy[i]), radius=r[i], color = color[i]))

    # if cylinder fit was successful then continue to fit atoms
    if moves < maxmoves:
        print('Successful cylinder fit!')

        # list of successful atom positions
        apos = []
        xy -= 0.5*l
        for i in range(natoms):
            try_pos = rng.random(3)*box
            try_pos -= box*0.5
            try_pos -= box * np.round(try_pos/box)
            atom_try = 0
            cyl_overlap = True
            # no atom overlap possible for the zeroth atom
            if i > 0:
                atom_overlap = True
                while (((cyl_overlap) or (atom_overlap)) and (atom_try < maxcount_atom)):
                    atom_try += 1
                    cyl_dists = distance(try_pos[:-1], xy, box[:-1])
                    cyl_overlap = np.any(cyl_dists < (r + 0.01))
                    atom_dists = distance(try_pos, np.array(apos), box)
                    atom_overlap = np.any(atom_dists < tol)
                    # if there is only atom overlap, try moving up in z first
                    if ((not cyl_overlap) and (atom_overlap)):
                        z_moves = 0
                        while ((z_moves < np.round(z/2.)) and (atom_overlap)):
                            z_moves += 0.5
                            try_pos[-1] += 0.5
                            try_pos -= box * np.round(try_pos/box)
                            atom_dists = distance(try_pos, np.array(apos), box)
                            atom_overlap = np.any(atom_dists < tol)

                    # otherwise if there is cylinder and atom overlap we need to try again
                    elif ((cyl_overlap) or (atom_overlap)):
                        try_pos = rng.random(3)*box - box*0.5
                        try_pos -= box * np.round(try_pos/box)
            else:
                atom_overlap = False
                while ((cyl_overlap) and (atom_try < maxcount_atom)):
                    atom_try += 1
                    cyl_dists = distance(try_pos[:-1], xy, box[:-1])
                    cyl_overlap = np.any(cyl_dists < (r + 0.01))
                    if cyl_overlap:
                        try_pos = rng.random(3)*box - box*0.5
                        try_pos -= box * np.round(try_pos/box)

            if ((not cyl_overlap) and (not atom_overlap)):
                print(f'atom {i} fit')
                apos.append(try_pos)

        if len(apos) == natoms:
            print(f'Successful atom fit, all {len(apos)} fit')
        else:
            print(f'Failed atom fit, only {len(apos)} fit :(')

        # map everything back to [0,l)
        apos = np.array(apos)
        apos += box*0.5
        xy += l*0.5

        # make some plots if you want
        if not noplots:
            fig, ax = plt.subplots()
            ax.set_xlim(0,l)
            ax.set_ylim(0,l)
            ax.set_aspect('equal')
            pos = np.array(apos)
            for i in range(n):
                ax.add_patch(plt.Circle((xy[i]), radius=r[i]))

            fig.savefig(outdir + '/holes.pdf')
            ax.scatter(pos[:,0],pos[:,1], s=2, color='k', zorder=99)
            fig.savefig(outdir + '/holes_with_atoms-2d.pdf')

            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #for xx,rr in zip(xy,r):
            #    Xc,Yc,Zc = data_for_cylinder_along_z(xx[0],xx[1],rr,z)
            #    ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
            #ax.scatter(apos[:,0], apos[:,1], apos[:,2], s=5., c='k')
            #fig.savefig(outdir + '/holes_with_atoms-3d.pdf')

            fig, ax= plt.subplots()
            ax.set_xlabel('Hole diameter')
            count, bins, _ = ax.hist(r*2.)
            xx = np.arange(mean_poresize-10., mean_poresize+10., 0.01)
            dist =  n/(std * np.sqrt(2 * np.pi))*np.exp(-(xx-mean_poresize)**2/(2*std**2))
            ax.plot(xx, dist, lw = 2, color='r')
            ax.set_xlim((bins.min()-1, bins.max()+1))
            fig.savefig(outdir + '/hole_size_distribution.pdf')

        # write out In.Holes and START file
        fo_name = outdir + '/START'
        with open(fo_name, 'w') as fo:
            fo.write(f'{natoms}\n{box[0]} {box[1]} {box[2]}')
            for p in apos:
                fo.write('\n'+' '.join([f'{pp:.2f}' for pp in p]))

        fo_name = outdir + '/In.Holes'
        with open(fo_name, 'w') as fo:
            fo.write('#python3 ' + ' '.join(sys.argv)+'\n')
            fo.write('#====================#\n')
            for a in vars(args):
                fo.write(f'#{a}= {getattr(args, a)}\n')
            fo.write('#====================#\n')
            fo.write(f'#number_holes= {n}\n')
            fo.write(f'#separation= {sep}\n')
            fo.write(f'#hole_fraction= {np.pi*(r**2).sum()/(l**2):.2f}\n')
            fo.write(f'#average_diameter= {r.mean()*2.:.2f}\n')
            fo.write('#====================#\n')
            for xx, rr in zip(xy,r):
                fo.write(f'cylinder= {xx[0]:.2f}  {xx[1]:.2f}  {rr:.2f}\n')

        # generate slurm run script
        if gen_slm:
            fo_name = outdir + '/job.slm'
            with open(fo_name, 'w') as fo:
                fo.write('#!/bin/tcsh\n')
                fo.write(f'#SBATCH --partition={partition}\n')
                fo.write(f'#SBATCH --time={simtime:02d}:00:00\n')
                fo.write(f'#SBATCH --nodes={nnodes}\n')
                fo.write('#SBATCH --no-requeue\n')
                fo.write('#SBATCH --export=NONE\n\n')
                fo.write(f'setenv OMP_NUM_THREADS {24*nnodes}\n\n')
                fo.write('set edip = /home/nigel/carbon/edip/2021/edip\n\n')
                fo.write('# Quenching Calculation\n')
                fo.write('/bin/cp START START_orig\n')
                fo.write(f'srun -n 1 -c {24*nnodes} $edip In.Quench > out\n')
                fo.write('/bin/cp ovito.xyz quench.xyz\n')
                fo.write('/bin/cp out out-quench\n')
                fo.write('# Annealing Calculation (breaking z-PBC)\n')
                fo.write('set natom = `head -1 ovito.xyz`\n')
                fo.write('printf  "$natom\\n100.0 100.0 50.0\\n" > START\n')
                fo.write('tail -$natom ovito.xyz | awk \'{print $3,$4,$5}\' >> START\n\n')
                fo.write(f'srun -n 1 -c {24*nnodes} $edip In.Anneal > out\n')

        # finally, tar the directory
        shutil.make_archive(outdir, 'gztar', outdir)
        shutil.rmtree(outdir)

    else:
        print('Failed cylinder fit :(')
        if not noplots:
            fig, ax = plt.subplots()
            ax.set_xlim(0,l)
            ax.set_ylim(0,l)
            ax.set_aspect('equal')
            for i in range(n):
                ax.add_patch(plt.Circle((xy[i]), radius=r[i]))
            fig.savefig('failed_holes.pdf')
        shutil.rmtree(outdir)
