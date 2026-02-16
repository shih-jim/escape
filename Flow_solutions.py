import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

class FlowSolutions:
    # Use the decorator to make the functions callable via Class name upon importing the py script
    @staticmethod
    def get_parker_wind(r, vs, rs):
        """
        Calculate the velocity profile of Isothermal Parker wind. Based on Cranmer (2004).
        
        Parameters
        ----------
        r: radius [m]
        vs: isothermal sound speed [m/s]
        rs: sonic point [m]

        Returns
        -------
        v: velocity profile [m/s]
        """

        f_r = 3 - 4 * rs / r # constant = -3 for critical solution
        brackets = -(rs / r)**4 * np.exp(f_r)
        v = np.zeros(r.size)

        #lambertw is the function W, brackets is the input z, and the result is the output w
        v[r <= rs] = np.sqrt(-np.real(lambertw(brackets[r <= rs], 0))) * vs 
        v[r > rs] = np.sqrt(-np.real(lambertw(brackets[r > rs], -1))) * vs
        
        return v
    
    @staticmethod
    def get_parker_wind_const(r, vs, rs, const):
        """
        Calculate the velocity profile of Isothermal Parker wind with a constant parameter. Based on Cranmer (2004).
        When the point of interest does not lie on the critical solution, a constant parameter is needed to adjust the solution.
        
        Parameters
        ----------
        r: radius [m]
        vs: isothermal sound speed [m/s]
        rs: sonic point [m]

        Returns
        -------
        v: velocity profile [m/s]
        """

        f_r = -const - 4.0 * rs / r
        brackets = -(rs / r)**4 * np.exp(f_r)
        v = np.zeros(r.size)

        v[r <= rs] = np.sqrt(-np.real(lambertw(brackets[r <= rs], 0))) * vs
        v[r > rs] = np.sqrt(-np.real(lambertw(brackets[r > rs], -1))) * vs

        return v
    
    @staticmethod
    def get_parker_wind_single(r,vs,rs):
        """
        Calculate the velocity of Isothermal Parker wind at a specificed radius r. Based on Cranmer (2004).

        Parameters
        ----------
        r: radius [m]
        vs: isothermal sound speed [m/s]
        rs: sonic point [m]

        Returns
        -------
        v: velocity [m/s]
        """    
    
        f_r = 3. - 4.*rs/r

        brackets = - (rs/r)**4.*np.exp(f_r)
    
        if (r <=rs):
            v = np.sqrt(-np.real(lambertw(brackets,0)))*vs
        else:
            v = np.sqrt(-np.real(lambertw(brackets,-1)))*vs
    
        return v
    
    @staticmethod
    def get_parker_wind_single_const(r,vs,rs,const):
        """
        Calculate the velocity of Isothermal Parker wind at a specificed radius r with a constant parameter. Based on Cranmer (2004).

        Parameters
        ----------
        r: radius [m]
        vs: isothermal sound speed [m/s]
        rs: sonic point [m]

        Returns
        -------
        v: velocity [m/s]
        """
        # as above but for single value of r
        
        # this function calculates the velocity structure (in cgs) of the isothermal Parker wind solution
        # c.f. Cranmer 2004
        
        f_r = -const - 4.*rs/r
        brackets = - (rs/r)**4.*np.exp(f_r)
        
        if (r <=rs):
            v = np.sqrt(-np.real(lambertw(brackets,0)))*vs
        else:
            v = np.sqrt(-np.real(lambertw(brackets,-1)))*vs
        
        return v
    
    