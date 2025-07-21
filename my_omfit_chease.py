try:
    # framework is running
    from omfit_classes.startup_choice import *
    # from omfit_classes.namelist import mod
except ImportError as _excp:
    # class is imported by itself
    if (
        'attempted relative import with no known parent package' in str(_excp)
        or 'No module named \'omfit_classes\'' in str(_excp)
        or "No module named '__main__.startup_choice'" in str(_excp)
    ):
        from startup_choice import *
    else:
        raise

from omfit_classes.omfit_ascii import OMFITascii
from omfit_classes.utils_math import cicle_fourier_smooth

import numpy as np
import scipy.interpolate as interpolate

__all__ = ['OMFITchease', 'OMFITcheaseLog', 'OMFITexptnz', 'OMFITnuplo']

class Constants:
    def __init__(self):
        self.mu_0 = 1.25663706E-6

constants = Constants()

def mod(a, b):
    return a % b

class OMFITchease(SortedDict, OMFITascii):
    r"""
    OMFIT class used to interface with CHEASE EXPEQ files

    :param filename: filename passed to OMFITascii class

    :param \**kw: keyword dictionary passed to OMFITascii class
    """

    def __init__(self, filename, **kw):
        SortedDict.__init__(self)
        OMFITascii.__init__(self, filename, **kw)
        self.dynaLoad = True

        self.rhotypes = {0: 'RHO_PSI', 1: 'RHO_TOR'}
        self.names = {
            1: ['PSI', 'PPRIME', 'FFPRIM'],
            2: ['PSI', 'PPRIME', 'JTOR'],
            3: ['PSI', 'PPRIME', 'JPAR'],
            41: ['RHO', 'PPRIME', 'FFPRIM'],
            42: ['RHO', 'PPRIME', 'JTOR'],
            43: ['RHO', 'PPRIME', 'JPAR'],
            44: ['RHO', 'PPRIME', 'IPAR'],
            45: ['RHO', 'PPRIME', 'Q'],
            81: ['RHO', 'PRES', 'FFPRIM'],
            82: ['RHO', 'PRES', 'JTOR'],
            83: ['RHO', 'PRES', 'JPAR'],
            84: ['RHO', 'PRES', 'IPAR'],
            85: ['RHO', 'PRES', 'Q'],
        }

    @dynaLoad
    def load(self):

        if self.filename is None or not os.stat(self.filename).st_size:
            return

        with open(self.filename, 'r') as f:
            lines = f.read().split('\n')

        tmp = lines[3].split()
        if len(tmp) == 1:
            self.version = 'standard'
            n1 = int(lines[3])
            n2 = 1
            n3 = 2
        elif len(tmp) == 3:
            self.version = 'mars'
            (
                n1,
                n2,
                n3,
            ) = list(map(int, tmp))
        else:
            raise OMFITexception('%s is not a valid OMFITchease file' % self.filename)

        self['EPS'] = float(lines[0])
        self['Z_AXIS'] = float(lines[1])
        self['P_SEP'] = float(lines[2])

        boundaries = []
        for b in range(n2):
            tmp = []
            for k, line in enumerate(lines[4 + n1 * b : 4 + n1 * (b + 1)]):
                tmp.append(list(map(float, line.split())))
            tmp = np.array(tmp)
            boundaries.append(tmp)

        bounds = ['PLASMA'] + ['LIMITER_%d' % k for k in range(len(boundaries) - 1)]
        self['BOUNDARY'] = {}
        for k, bound in enumerate(bounds):
            self['BOUNDARY'][bound] = {}
            self['BOUNDARY'][bound]['R'] = boundaries[k][:, 0]
            self['BOUNDARY'][bound]['Z'] = boundaries[k][:, 1]
            if n3 > 2:
                self['BOUNDARY'][bound]['x'] = boundaries[k][:, 2]

        offset = 4 + n1 * n2
        if self.version == 'standard':
            tmp = lines[offset].split()
            l = int(tmp[0])
            if len(tmp) > 1:
                self.nppfun = int(tmp[1])
            else:
                self.nppfun = 4
            tmp = lines[offset + 1].split()
            self.nsttp = int(tmp[0])
            self.nrhotype = int(tmp[1])
            self['MODE'] = self.nppfun * 10 + self.nsttp
        elif self.version == 'mars':
            l = int(lines[offset])
            self['MODE'] = int(lines[offset + 1])

        for k, name in enumerate(self.names[self['MODE']]):
            if (name == 'RHO') and (self.version == 'standard'):
                name = self.rhotypes[self.nrhotype]
            self[name] = np.array(list(map(float, lines[offset + 2 + l * k : offset + 2 + l * (k + 1)])))
        offset = offset + 2 + l * (k + 1) + 1

        self['NOTES'] = []
        for line in lines[offset:]:
            if line.strip():
                self['NOTES'].append(line.rstrip('\n'))
        self['NOTES'] = '\n'.join(self['NOTES'])



    @dynaSave
    def save(self):
        """
        Method used to save the content of the object to the file specified in the .filename attribute

        :return: None
        """

        with open(self.filename, 'w') as f:
            f.write(str(self['EPS']) + '\n')
            f.write(str(self['Z_AXIS']) + '\n')
            f.write(str(self['P_SEP']) + '\n')
            n1 = len(self['BOUNDARY']['PLASMA']['R'])
            if self.version == 'standard':
                n2 = 1
                n3 = 2
                f.write(str(n1) + '\n')
            elif self.version == 'mars':
                n2 = len(self['BOUNDARY'])
                n3 = len(self['BOUNDARY']['PLASMA'])
                tmp = np.hstack((n1, n2, n3))
                f.write(' '.join(map(str, tmp)) + '\n')
            bounds = ['PLASMA'] + ['LIMITER_%d' % k for k in range(n2 - 1)]
            for bound in bounds:
                tmp = np.vstack([self['BOUNDARY'][bound]['R'], self['BOUNDARY'][bound]['Z']]).T
                shape = list(tmp.shape)
                for k in range(tmp.shape[0]):
                    f.write(' '.join(['%5.9e' % x for x in tmp[k, :]]) + '\n')
            # Write length of profiles
            l = len(self[self.names[self['MODE']][-1]])
            if self.version == 'standard':
                tmp = np.hstack((l, self.nppfun))
                f.write(' '.join(map(str, tmp)) + '\n')
                tmp = np.hstack((self.nsttp, self.nrhotype))
                f.write(' '.join(map(str, tmp)) + '\n')
            elif self.version == 'mars':
                f.write(str(l) + '\n')
                # Write number corresponding to profile combitation
                f.write(str(self['MODE']) + '\n')
            for name in self.names[self['MODE']]:
                if (name == 'RHO') and (self.version == 'standard'):
                    name = self.rhotypes[self.nrhotype]
                tmp = self[name]
                shape = list(tmp.shape)
                for k in range(shape[0]):
                    f.write(str('%5.9e' % tmp[k]) + '\n')
            if len(self['NOTES'].strip()):
                f.write(self['NOTES'].rstrip('\n') + '\n')



    @staticmethod
    def splineRZ(R, Z, Nnew):
        """
        Auxilliary function to spline single boundary from EXPEQ

        :param R: array 1 (R coordinate)

        :param Z: array 2 (Z coordinate)

        :param Nnew: new number of points

        :return: smoothed R,Z

        """
        npoints = len(R)
        degree = 3
        t = np.linspace(-np.pi, np.pi, npoints)
        ipl_t = np.linspace(-np.pi, np.pi, Nnew)
        R_i = interpolate.UnivariateSpline(t, R, k=degree, s=1.0e-5)(ipl_t)
        Z_i = interpolate.UnivariateSpline(t, Z, k=degree, s=1.0e-5)(ipl_t)
        return R_i, Z_i



    def EQsmooth(self, keep_M_harmonics, inPlace=True, equalAngle=False, doPlot=False):
        """
        smooth plasma boundary by zeroing out high harmonics

        :param keep_M_harmonics: how many harmonics to keep

        :param inPlace: operate in place (update this file or not)

        :param equalAngle: use equal angle interpolation, and if so, how many points to use

        :param doPlot: plot plasma boundary before and after

        :return: smoothed R and Z coordinates
        """

        R = self['BOUNDARY']['PLASMA']['R']
        Z = self['BOUNDARY']['PLASMA']['Z']

        RS, ZS = cicle_fourier_smooth(R, Z, keep_M_harmonics, equalAngle=equalAngle, doPlot=doPlot)

        if inPlace:
            self['BOUNDARY']['PLASMA']['R'] = RS
            self['BOUNDARY']['PLASMA']['Z'] = ZS

        return RS, ZS



    def addLimiter(self, R, Z):
        """
        Insertion of a wall defined by coordinates (R,Z)

        :R = radial coordinate

        :Z = vertical coordinate

        Note: both must be normalized, and ndarray type
        """
        if len(self['BOUNDARY']) == 0:
            raise ("Error. No PLASMA found")
        else:
            d = len(self['BOUNDARY']) - 1
            self['BOUNDARY']['LIMITER_' + str(d)] = {}
            self['BOUNDARY']['LIMITER_' + str(d)]['R'] = R
            self['BOUNDARY']['LIMITER_' + str(d)]['Z'] = Z



    def modifyBoundary(self):
        """
        Interactively modify plasma boundary
        """
        R = self['BOUNDARY']['PLASMA']['R']
        Z = self['BOUNDARY']['PLASMA']['Z']
        tmp = fluxGeo(R, Z)

        bs = BoundaryShape(
            a=tmp['a'],
            eps=tmp['eps'],
            kapu=tmp['kapu'],
            kapl=tmp['kapl'],
            delu=tmp['delu'],
            dell=tmp['dell'],
            zetaou=tmp['zetaou'],
            zetaiu=tmp['zetaiu'],
            zetail=tmp['zetail'],
            zetaol=tmp['zetaol'],
            zoffset=tmp['zoffset'],
            rbbbs=R,
            zbbbs=Z,
            upnull=False,
            lonull=True,
        )

        self['BOUNDARY']['PLASMA']['R'] = bs['r']
        self['BOUNDARY']['PLASMA']['Z'] = bs['z']

        bs.plot()



    def from_gEQDSK(self, gEQDSK=None, conformal_wall=False, mode=None, rhotype=0, version=None, cocos=2):
        """
        Modify CHEASE EXPEQ file with data loaded from gEQDSK

        :param gEQDSK: input gEQDKS file from which to copy the plasma boundary

        :param conformal_wall: floating number that multiplies plasma boundary (should be >1)

        :param mode: select profile to use from gEQDSK

        :param rhotype: 0 for poloidal flux. 1 for toroidal flux. Only with version=='standard'

        :param version: either 'standard' or 'mars'

        """

        if version is None:
            if hasattr(self, 'version'):
                version = self.version
            else:
                version = 'standard'
        if mode is None:
            if 'MODE' in self:
                mode = self['MODE']
            else:
                mode = {'standard': 41, 'mars': 1}[version]

        gEQDSK = gEQDSK.cocosify(cocos, True, True, False)

        R0 = gEQDSK['RCENTR']
        B0 = gEQDSK['BCENTR']

        # normalizations from section 5.4.6 of CHEASE manual

        if version not in ['standard', 'mars']:
            OMFITexception('Cannot create OMFITchease in version %s' % version)
        else:
            self.version = version

        self['EPS'] = gEQDSK['fluxSurfaces']['geo']['eps'][-1]
        self['Z_AXIS'] = gEQDSK['ZMAXIS'] / R0
        self['P_SEP'] = gEQDSK['PRES'][-1] / (B0**2 / constants.mu_0)

        self['BOUNDARY'] = {}
        self['BOUNDARY']['PLASMA'] = {}
        self['BOUNDARY']['PLASMA']['R'] = gEQDSK['RBBBS'] / R0
        self['BOUNDARY']['PLASMA']['Z'] = gEQDSK['ZBBBS'] / R0

        if version == 'mars':
            # conformal wall
            if 'LIMITER_0' not in self['BOUNDARY']:
                self['BOUNDARY']['LIMITER_0'] = {}
            if conformal_wall:
                self['BOUNDARY']['LIMITER_0']['R'] = (self['BOUNDARY']['PLASMA']['R'] - 1) * conformal_wall + 1
                self['BOUNDARY']['LIMITER_0']['Z'] = (self['BOUNDARY']['PLASMA']['Z'] - self['Z_AXIS']) * conformal_wall + self['Z_AXIS']
            # original wall
            else:
                self['BOUNDARY']['LIMITER_0']['R'] = gEQDSK['RLIM'] / R0
                self['BOUNDARY']['LIMITER_0']['Z'] = gEQDSK['ZLIM'] / R0

        # Deleting all possible profiles to avoid conflicts
        profiles = ['PSI', 'RHO_PSI', 'RHO_TOR']  # coordinates
        profiles += ['PPRIME', 'PRES']  # pressure
        profiles += ['FFPRIM', 'JTOR', 'JPAR', 'IPAR', 'Q']  # current
        for prof in profiles:
            self.safe_del(prof)

        if version == 'standard':

            # coordinate
            if rhotype == 0:
                self['RHO_PSI'] = gEQDSK['AuxQuantities']['RHOp']
            elif rhotype == 1:
                self['RHO_TOR'] = gEQDSK['AuxQuantities']['RHO']
            self.nrhotype = rhotype

            # pressure
            if mode in [1, 2] or int(mode / 10) == 4:
                self['PPRIME'] = gEQDSK['PPRIME'] / np.abs(B0 / (constants.mu_0 * R0**2))
            elif int(mode / 10) == 8:
                self['PRES'] = gEQDSK['PRES'] / (B0**2) * constants.mu_0
            else:
                raise OMFITexception('Cannot define pressure for MODE %d' % mode)

            # current/q
            if mod(mode, 10) == 1:
                self['FFPRIM'] = gEQDSK['FFPRIM'] / np.abs(B0)
            elif mod(mode, 10) == 2:
                self['JTOR'] = (
                    gEQDSK['fluxSurfaces']['avg']['Jt/R'] / gEQDSK['fluxSurfaces']['avg']['1/R'] / np.abs(B0 / (R0 * constants.mu_0))
                )
            elif mod(mode, 10) == 3:
                raise OMFITexception('%s cannot be loaded from gEQDSK yet' % self.names[mode][2])
            elif mod(mode, 10) == 4:
                raise OMFITexception('%s cannot be loaded from gEQDSK yet' % self.names[mode][2])
            elif mod(mode, 10) == 5:
                self['Q'] = gEQDSK['QPSI']
            else:
                raise OMFITexception('Cannot define current or q for MODE %d' % mode)

        elif version == 'mars':
            self['PSI'] = gEQDSK['AuxQuantities']['RHOp']
            self['PPRIME'] = gEQDSK['PPRIME'] / np.abs(B0 / (constants.mu_0 * R0**2))
            if mode == 1:
                self['FFPRIM'] = gEQDSK['FFPRIM'] / np.abs(B0)
            elif mode == 2:
                self['JTOR'] = (
                    gEQDSK['fluxSurfaces']['avg']['Jt/R'] / gEQDSK['fluxSurfaces']['avg']['1/R'] / np.abs(B0 / (R0 * constants.mu_0))
                )
            elif mode == 3:
                raise OMFITexception('%s cannot be loaded from gEQDSK yet' % self.names[mode][2])
            else:
                raise OMFITexception('MODE %s unknown or not supported' % mode)

        self['MODE'] = mode
        self['NOTES'] = ''
        self.version = version
        return self



    def plot(self, bounds=None, **kw):
        """

        :param bounds:
        :param kw:
        :return:
        """
        from matplotlib import pyplot

        if bounds is None:
            bounds = self['BOUNDARY']

        kw0 = copy.copy(kw)
        if 'lw' in kw:
            kw0['lw'] = kw0['lw'] + 1
        elif 'linewidth' in kw:
            kw0['linewidth'] = kw0['linewidth'] + 1
        else:
            kw0['lw'] = 2
        kw0['color'] = 'k'

        if self.version == 'standard':
            X = self[self.rhotypes[self.nrhotype]]
            Xlabel = '$\\rho$'
        elif self.version == 'mars':
            X = self['PSI']
            Xlabel = '$\\sqrt{\\psi}$'

        ax = pyplot.subplot(2, 2, 2)
        quantity = self.names[self['MODE']][1]
        ax.plot(X, self[quantity], **kw)
        ax.set_title(quantity)
        color = ax.lines[-1].get_color()
        kw['color'] = color

        ax = pyplot.subplot(2, 2, 4)
        quantity = self.names[self['MODE']][2]
        ax.plot(X, self[quantity], **kw)
        ax.set_title(quantity)
        ax.set_xlabel(Xlabel)

        ax = pyplot.subplot(1, 2, 1)
        for bound in self['BOUNDARY']:
            if bound == 'PLASMA':
                ax.plot(self['BOUNDARY'][bound]['R'], self['BOUNDARY'][bound]['Z'], **kw)
            else:
                ax.plot(self['BOUNDARY'][bound]['R'], self['BOUNDARY'][bound]['Z'], **kw0)
        ax.set_aspect('equal')
        ax.set_frame_on(False)




class OMFITcheaseLog(SortedDict, OMFITascii):
    r"""
    OMFIT class used to parse the CHEASE log FILES for the following parameters:
    betaN, NW, CURRT, Q_EDGE, Q_ZERO, R0EXP, B0EXP, Q_MIN, S_Q_MIN, Q_95

    :param filename: filename passed to OMFITascii class

    :param \**kw: keyword dictionary passed to OMFITascii class
    """

    def __init__(self, filename, **kw):
        SortedDict.__init__(self)
        OMFITascii.__init__(self, filename, **kw)
        self.dynaLoad = True


    @dynaLoad
    def load(self):
        """
        Load CHEASE log file data
        """
        # Array with string pattern, variable name and position of the number to store
        mystring = [
            ('CSV=', 'NW', -1),
            ('GEXP', 'BetaN', -1),
            ('TOTAL CURRENT -->', 'CURRT', 0),
            ('Q_EDGE', 'Q_EDGE', 0),
            ('Q_ZERO', 'Q_ZERO', 0),
            ('R0 [M]', 'R0EXP', 0),
            ('B0 [T]', 'B0EXP', 0),
            ('MINIMUM Q VALUE', 'Q_MIN', -1),
            ('S VALUE OF QMIN', 'S_Q_MIN', -1),
            ('Q AT 95%', 'Q_95', -1),
            ('RESIDU', 'RESIDUAL', 2),
        ]

        with open(self.filename, 'r') as f:
            for line in f.readlines():
                for word, name, pos in mystring:
                    str_found = line.find(word)
                    if (word == 'CSV=') and (name not in self):
                        self[name] = []
                    if str_found != -1:
                        if word == 'CSV=':
                            try:
                                self[name].append(ast.literal_eval(line.split()[pos]))
                            except SyntaxError:
                                tmp = re.findall(r'(\w+=)(\d+)', line.split()[pos])[0]
                                self[name].append(ast.literal_eval(tmp[-1]))
                        elif (word != 'RESIDU') or ('EPSLON' in line.split()):
                            self[name] = ast.literal_eval(line.split()[pos])
        return self



    def read_VacuumMesh(self, nv):
        """
        Read vacuum mesh from CHEASE log file
        :param nv: number of radial intervals in vacuum (=NV from input namelist)
        """
        self['VACUUM MESH'] = {}
        NW = []
        rw = []
        mystring = 'VACUUM MESH CSV ='
        with open(self.filename, 'r') as f:
            data = f.readlines()
            for line_no, line in enumerate(data):
                if line.find(mystring) != -1:
                    break
            for ii in range(nv + 1):
                NW.append(int(data[line_no + ii + 1].split()[0]))
                rw.append(float(data[line_no + ii + 1].split()[1]))
        self['VACUUM MESH']['NW'] = np.asarray(NW)
        self['VACUUM MESH']['rw'] = np.asarray(rw)
        return self




class OMFITexptnz(SortedDict, OMFITascii):
    r"""
    OMFIT class used to interface with EXPTNZ files containing kinetic profiles

    :param filename: filename passed to OMFITascii class

    :param \**kw: keyword dictionary passed to OMFITascii class
    """

    def __init__(self, filename, **kw):
        SortedDict.__init__(self)
        OMFITascii.__init__(self, filename, **kw)
        self.dynaLoad = True


    @dynaLoad
    def load(self):
        with open(self.filename, 'r') as f:
            header = f.readline().split()
        if not len(header):
            return
        npoints = int(header[0])
        tmp = np.loadtxt(self.filename, skiprows=1)
        nprofs = int(len(tmp) / npoints)
        for k in range(nprofs):
            if ',' in header[k + 1]:
                name = header[k + 1][:-1]
            else:
                name = header[k + 1]
            if name == 'rhopsi':
                rhopsi = tmp[: (k + 1) * npoints]
                self['rhopsi'] = xarray.DataArray(
                    rhopsi, dims=['rhopsi'], coords={'rhopsi': rhopsi}, attrs={'Description': 'Normalized poloidal flux'}
                )
                continue
            else:
                tmp2 = tmp[(k) * npoints : (k + 1) * npoints]
                self[name] = xarray.DataArray(tmp2, dims=['rhopsi'], coords={'rhopsi': self['rhopsi']})



    def exptnz2mars(self):
        from omfit_classes.omfit_mars import OMFITmarsProfile

        outprofs = {}
        for item in self:
            if item == 'rhopsi':
                continue
            proffile = item + '_prof'
            with open(proffile, 'w') as f:
                tmp = np.vstack(list([self['rhopsi'], self[item]])).T
                shape = list(tmp.shape)
                shape[1] = shape[1] - 1
                f.write(' '.join(map(str, shape)) + '\n')
                for k in range(tmp.shape[0]):
                    f.write(' '.join(['%5.9e' % x for x in tmp[k, :]]) + '\n')
            outprofs[item] = OMFITmarsProfile(proffile)
        return outprofs




class OMFITnuplo(SortedDict, OMFITascii):
    """CHEASE NUPLO output file"""

    def __init__(self, filename, **kw):
        OMFITascii.__init__(self, filename, **kw)
        SortedDict.__init__(self)
        self.dynaLoad = True


    @dynaLoad
    def load(self):
        if self.filename is None or not os.stat(self.filename).st_size:
            return

        def naneval(val):
            if 'nan' not in val.lower():
                return eval(val)
            else:
                printw("  ---> WARNING!! NAN ENCOUNTERED!!!")
                return np.NaN

        def readlines_size(lines, nsize):
            out = []
            linesread = 0
            while len(out) < nsize:
                out += list(map(naneval, lines[linesread].split()))
                linesread += 1
            if len(out) == nsize:
                return np.array(out), linesread
            elif len(out) > nsize:
                printw("  ---> WARNING!! UNEXPECTED LINES ENCOUNTERED!!!")
                return np.array(out), linesread

        with open(self.filename, mode='r') as f:
            lines = f.read().splitlines()

        NUMS = self['NUMS'] = {}
        DATA = self['DATA'] = {}

        [
            NUMS['insur'],
            NUMS['nchi'],
            NUMS['nchi1'],
            NUMS['npsi'],
            NUMS['npsi1'],
            NUMS['ns'],
            NUMS['ns1'],
            NUMS['nt'],
            NUMS['nt1'],
            NUMS['ins'],
            NUMS['inr'],
            NUMS['inbchi'],
            NUMS['intext'],
        ] = list(map(naneval, lines[0].split()))

        [
            NUMS['ncurv'],
            NUMS['nmesha'],
            NUMS['nmeshb'],
            NUMS['nmeshc'],
            NUMS['nmeshd'],
            NUMS['nmeshe'],
            NUMS['npoida'],
            NUMS['npoidb'],
            NUMS['npoidc'],
            NUMS['npoidd'],
            NUMS['npoide'],
            NUMS['niso'],
            NUMS['nmgaus'],
        ] = list(map(naneval, lines[1].split()))

        NUMS['niso1'] = NUMS['niso'] + 1
        offset = 2 + NUMS['intext']

        self['nuplo_text'] = '\n'.join(lines[2:offset])

        DATA['iball'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i
        DATA['imerci'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i
        DATA['imercr'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i

        tmp, i = readlines_size(lines[offset:], 10)
        offset += i
        [
            NUMS['solpda'],
            NUMS['sopldb'],
            NUMS['solpdc'],
            NUMS['solpdd'],
            NUMS['solpde'],
            NUMS['zrmax'],
            NUMS['zrmin'],
            NUMS['zzmax'],
            NUMS['zzmin'],
            NUMS['pangle'],
        ] = tmp

        DATA['aplace'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['awidth'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['bplace'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['bwidth'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['cplace'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['cwidth'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['dplace'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['dwidth'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['eplace'], i = readlines_size(lines[offset:], 10)
        offset += i
        DATA['ewidth'], i = readlines_size(lines[offset:], 10)
        offset += i

        DATA['ztet'], i = readlines_size(lines[offset:], NUMS['nt1'])
        offset += i
        DATA['csig'], i = readlines_size(lines[offset:], NUMS['ns1'])
        offset += i
        DATA['cs'], i = readlines_size(lines[offset:], NUMS['niso1'])
        offset += i
        # DATA['COMMENTS']['cs'] = 's-mesh ~ sqrt(psi_normalized)'
        DATA['zchi'], i = readlines_size(lines[offset:], NUMS['nchi1'])
        offset += i
        # DATA['COMMENTS']['zchi'] = 'chi-mesh (note: meaning depends on Jacobian)'
        DATA['zcsipr'], i = readlines_size(lines[offset:], NUMS['niso1'])
        offset += i
        DATA['zrtet'], i = readlines_size(lines[offset:], NUMS['nt'])
        offset += i
        DATA['zztet'], i = readlines_size(lines[offset:], NUMS['nt'])
        offset += i
        DATA['zrsur'], i = readlines_size(lines[offset:], NUMS['insur'])
        offset += i
        DATA['ztsur'], i = readlines_size(lines[offset:], NUMS['insur'])
        offset += i
        DATA['zzsur'], i = readlines_size(lines[offset:], NUMS['insur'])
        offset += i
        # DATA['COMMENTS']['zrsur_zzsur'] = '(R-Rmag,Z-Zmag) values plasma boundary surface'
        DATA['zabis'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zabit'], i = readlines_size(lines[offset:], NUMS['nt1'])
        offset += i
        DATA['zabic'], i = readlines_size(lines[offset:], NUMS['nchi1'])
        offset += i
        DATA['zoart'], i = readlines_size(lines[offset:], NUMS['nt1'])
        offset += i
        DATA['zabipr'], i = readlines_size(lines[offset:], NUMS['niso1'])
        offset += i
        DATA['zabisg'], i = readlines_size(lines[offset:], NUMS['ns1'])
        offset += i
        DATA['zabsm'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zabr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zoqs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zoqr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zodqs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zodqr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i

        DATA['zoshs'], i = readlines_size(lines[offset:], NUMS['npsi1'] + 1)
        offset += i
        DATA['zoshs'] = DATA['zoshs'][:-1]  # bad number at the end of this line...

        DATA['zoshr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zojbs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zojbr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i

        DATA['zojbss'] = np.zeros((NUMS['ins'], 4))
        DATA['zojbss'][:, 0], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zojbss'][:, 1], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zojbss'][:, 2], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zojbss'][:, 3], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i

        DATA['zojbsr'] = np.zeros((NUMS['inr'], 4))
        DATA['zojbsr'][:, 0], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zojbsr'][:, 1], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zojbsr'][:, 2], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zojbsr'][:, 3], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i

        DATA['zojps'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zojpr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zotrs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zotrr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zohs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zodis'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zodrs'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zopps'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zoppr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zops'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zopr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zotts'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zottr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zots'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zotr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zoips'], i = readlines_size(lines[offset:], NUMS['ins'])
        offset += i
        DATA['zoipr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zobetr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zobets'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i
        DATA['zofr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zoars'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i
        DATA['zojr'], i = readlines_size(lines[offset:], NUMS['inr'])
        offset += i
        DATA['zabs'], i = readlines_size(lines[offset:], NUMS['npsi1'])
        offset += i

        DATA['rriso'] = np.zeros((NUMS['nmgaus'] * NUMS['nt1'], NUMS['npsi1']))
        DATA['rziso'] = np.zeros((NUMS['nmgaus'] * NUMS['nt1'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['rriso'][:, j], i = readlines_size(lines[offset:], NUMS['nmgaus'] * NUMS['nt1'])
            offset += i
            DATA['rziso'][:, j], i = readlines_size(lines[offset:], NUMS['nmgaus'] * NUMS['nt1'])
            offset += i

        DATA['rrcurv'], i = readlines_size(lines[offset:], NUMS['ncurv'])
        offset += i
        DATA['rzcurv'], i = readlines_size(lines[offset:], NUMS['ncurv'])
        offset += i
        # DATA['COMMENTS']['rrcurv_rzcurv'] = '(R-Rmag, Z-Zmag) values of zero curvature line, as from NUPLO'

        aa, i = readlines_size(lines[offset:], NUMS['inbchi'] * NUMS['npsi1'])
        offset += i
        ij = 0
        DATA['zrchi'] = np.zeros((NUMS['inbchi'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['zrchi'][:, j] = aa[ij : ij + NUMS['inbchi']]
            ij += NUMS['inbchi']
        aa, i = readlines_size(lines[offset:], NUMS['inbchi'] * NUMS['npsi1'])
        offset += i
        ij = 0
        DATA['zzchi'] = np.zeros((NUMS['inbchi'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['zzchi'][:, j] = aa[ij : ij + NUMS['inbchi']]
            ij += NUMS['inbchi']
        # DATA['COMMENTS']['zrchi_zzchi'] = '(R-Rmag, Z-Zmag) values of chi=cst lines'

        aa, i = readlines_size(lines[offset:], NUMS['nchi'] * NUMS['npsi1'])
        offset += i
        ij = 0
        DATA['rshear'] = np.zeros((NUMS['nchi'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['rshear'][:, j] = aa[ij : ij + NUMS['nchi']]
            ij += NUMS['nchi']
        aa, i = readlines_size(lines[offset:], NUMS['nchi'] * NUMS['npsi1'])
        offset += i
        ij = 0
        DATA['cr'] = np.zeros((NUMS['nchi'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['cr'][:, j] = aa[ij : ij + NUMS['nchi']]
            ij += NUMS['nchi']
        aa, i = readlines_size(lines[offset:], NUMS['nchi'] * NUMS['npsi1'])
        offset += i
        ij = 0
        DATA['cz'] = np.zeros((NUMS['nchi'], NUMS['npsi1']))
        for j in np.arange(NUMS['npsi1']):
            DATA['cz'][:, j] = aa[ij : ij + NUMS['nchi']]
            ij += NUMS['nchi']

        DATA['crp'] = np.vstack((DATA['cr'], DATA['cr'][0, :]))
        DATA['czp'] = np.vstack((DATA['cz'], DATA['cz'][0, :]))
        # DATA['COMMENTS']['cr_cz_p'] = "(R-Rmag, Z-Zmag) values of psi=cst surfaces, 'p' for including periodic point"

        DATA['rshearp'] = np.vstack((DATA['rshear'], DATA['rshear'][0, :]))
        # DATA['COMMENTS']['rshear_p'] = "(R-Rmag, Z-Zmag) values of local shear S, 'p' for including periodic point"

        DATA['r0curv'] = np.hstack((DATA['rrcurv'][-2::-2], DATA['rrcurv'][1::2]))
        DATA['z0curv'] = np.hstack((DATA['rzcurv'][-2::-2], DATA['rzcurv'][1::2]))

        # DATA['COMMENTS']['r0curv, z0curv'] = "(R-Rmag, Z-Zmag) values of zero curvature line ordered from top to bottom"


# ############################################
# if '__main__' == __name__:
#     test_classes_main_header()

#     tmp1 = OMFITchease(OMFITsrc + '/../samples/EXPEQ.OUT')

#     tmp1.plot()
#     tmp1.EQsmooth(10)
#     tmp1.plot(bounds=['PLASMA'])

#     from matplotlib import pyplot

#     pyplot.show()