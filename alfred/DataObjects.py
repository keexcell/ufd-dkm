from alfred import utils

class Data():
    def __init__(self, data):
        self.data = data

    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return Data(new_data)

class Band():
    def __init__(self, flux, fluxerr, name):
        self.flux = flux
        self.fluxerr = fluxerr
        self.mag = utils.flux2mag(flux)
        self.magerr = utils.fluxerr2magerr(flux, fluxerr)
        self.str = name

class LSSTData(Data):
    def __init__(self, data, lsst_release, tract):
        super(LSSTData, self).__init__(data)
        self.lsst_survey = lsst_release
        self.tract = tract
        self.field = utils.get_field(tract)
        
        ## coordinates
        self.ra_limits = (data['coord_ra'].min(), data['coord_ra'].max())
        self.dec_limits = (data['coord_dec'].min(), data['coord_dec'].max())
        self.rubin_ra = data['coord_ra']
        self.rubin_dec = data['coord_dec']

        ## Rubin bands
        self.g = Band(data['g_psfFlux'], data['g_psfFlux'], 'g')
        self.r = Band(data['r_psfFlux'], data['r_psfFlux'], 'r')
        self.i = Band(data['i_psfFlux'], data['i_psfFlux'], 'i')
        self.z = Band(data['z_psfFlux'], data['z_psfFlux'], 'z')
        #then because I have so many functions already defined, some retroactive definitions:
        self.g_mag = self.g.mag
        self.g_magerr = self.g.magerr
        self.r_mag = self.r.mag
        self.r_magerr = self.r.magerr
        self.i_mag = self.i.mag
        self.i_magerr = self.i.magerr
        self.z_mag = self.z.mag
        self.z_magerr = self.z.magerr

    ## morphology
    def band_psfmincmodel(band, self):
        psf_flux = utils.flux2mag(self.data[f'{band}_psfFlux'])
        cmodel_flux = utils.flux2mag(self.data[f'{band}_cModelFlux'])
        return psf_flux - cmodel_flux
    def band_psfdivcmodel(band, self):
        psf_flux = utils.flux2mag(self.data[f'{band}_psfFlux'])
        cmodel_flux = utils.flux2mag(self.data[f'{band}_cModelFlux'])
        return psf_flux / cmodel_flux
        
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return LSSTData(new_data, self.lsst_release, self.tract)


class LSSTnEuclidData(LSSTData):
    def __init__(self, data, lsst_survey, euclid_survey, field):
        super(LSSTnEuclidData, self).__init__(data, lsst_survey, field)
        self.euclid_survey = euclid_survey
        
        ## coordinates
        self.euclid_ra = data['right_ascension']
        self.euclid_dec = data['declination']
    
        ## Euclid bands (flux given in mu_Jy)
        num = 2 #as suggested in Zerjal et al
        #convert fluxes to nJy, that's what the flux -> mag functions assume
        self.VIS = Band(data[f'FLUX_VIS_{num}FWHM_APER'.lower()]*(10**3), 
                        data[f'FLUXERR_VIS_{num}FWHM_APER'.lower()]*(10**3),
                        'VIS')
        self.H = Band(data[f'FLUX_H_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_H_{num}FWHM_APER'.lower()]*(10**3),
                      'H')
        self.Y = Band(data[f'FLUX_Y_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_Y_{num}FWHM_APER'.lower()]*(10**3),
                      'Y')
        self.J = Band(data[f'FLUX_J_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_J_{num}FWHM_APER'.lower()]*(10**3),
                      'J')
        #then because I have so many functions already defined, some retroactive definitions:
        self.VIS_mag = self.VIS.mag
        self.VIS_magerr = self.VIS.magerr
        self.H_mag = self.H.mag
        self.H_magerr = self.H.magerr
        self.Y_mag = self.Y.mag
        self.Y_magerr = self.Y.magerr
        self.J_mag = self.J.mag
        self.J_magerr = self.J.magerr
        
        ## morphology
        self.pointlikeprob = data['point_like_prob']
        self.ellipticity = data['ellipticity']
        self.mumax_minus_mag = self.data['mumax_minus_mag']
        
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return LSSTnEuclidData(new_data, self.lsst_survey, self.euclid_survey, self.tract)
