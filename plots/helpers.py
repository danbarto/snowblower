#Stolen from Daniel Spitzbart at https://github.com/danbarto/tW_scattering/blob/master/plots/helpers.py

import numpy as np
import os
import matplotlib.pyplot as plt
from coffea import hist
from yahist import Hist1D, Hist2D
import shutil


import re


colors = {
    'tW_scattering': '#FF595E',
    'topW_v2': '#FF595E',
    'topW_v3': '#FF595E',
    'signal': '#FF595E',
    #'tW_scattering': '#000000', # this would be black
    'TTW': '#8AC926',
    'prompt': '#8AC926',
    'TTX': '#FFCA3A',
    'TTZ': '#FFCA3A',
    'lost lepton': '#FFCA3A',
    'TTH': '#34623F',
    'TTTT': '#0F7173',
    'ttbar': '#1982C4',
    'non prompt': '#1982C4',
    'wjets': '#6A4C93',
    'diboson': '#525B76',
    'rare': '#6A4C93',
    'WZ': '#525B76',
    'WW': '#34623F',
    'DY': '#6A4C93',
    'charge flip': '#6A4C93',
    'MuonEG': '#000000',
}
'''
other colors (sets from coolers.com):
#525B76 (gray)
#34623F (hunter green)
#0F7173 (Skobeloff)
'''

my_labels = {
    'tW_scattering': 'top-W scat.',
    'topW_v2': 'top-W scat.',
    'topW_v3': 'top-W scat.',
    'signal': 'top-W scat.',
    'prompt': 'prompt/irred.',
    'non prompt': 'nonprompt',
    'charge flip': 'charge flip',
    'lost lepton': 'lost lepton',
    'TTW': r'$t\bar{t}$W+jets',
    'TTX': r'$t\bar{t}$Z/H',
    'TTH': r'$t\bar{t}$H',
    'TTZ': r'$t\bar{t}$Z',
    'TTTT': r'$t\bar{t}t\bar{t}$',
    'ttbar': r'$t\bar{t}$+jets',
    'wjets': 'W+jets',
    'DY': 'Drell-Yan',
    'diboson': 'VV/VVV',
    'rare': 'Rare',
    'WZ': 'WZ',
    'WW': 'WW',
    'MuonEG': 'Observed',
    'pseudodata': 'Pseudo-data',
    'uncertainty': 'Uncertainty',
}


data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

signal_err_opts = {
    'linestyle':'-',
    'color':'crimson',
    'elinewidth': 1,
}

#signal_err_opts = {
#    'linestyle': '-',
#    'marker': '.',
#    'markersize': 0.,
#    'color': 'k',
#    'elinewidth': 1,
#    'linewidth': 2,
#}


error_opts = {
    'label': 'uncertainty',
    'hatch': '///',
    'facecolor': 'none',
    'edgecolor': (0,0,0,.5),
    'linewidth': 0
}

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 1.0
}

line_opts = {
    'linestyle':'-',
    'linewidth': 3,
}

signal_fill_opts = {
    'linewidth': 2,
    'linecolor': 'k',
    'edgecolor': (1,1,1,0.0),
    'facecolor': 'none',
    'alpha': 0.1
}

import mplhep as hep
plt.style.use(hep.style.CMS)

def finalizePlotDir( path ):
    path = os.path.expandvars(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copy( os.path.expandvars( '$TWHOME/tools/php/index.php' ), path )

def makePlot(output, histo, axis, bins=None, data=[], normalize=True, log=False, save=False, axis_label=None, ratio_range=None, upHists=[], downHists=[], shape=False, ymax=False, new_colors=colors, new_labels=my_labels, order=None, signals=[], omit=[], lumi=60.0, binwnorm=None, overlay=None, is_data=True, y_axis_label='Events', rescale={}, obs_label='Observation'):
    
    #if save:
    #    finalizePlotDir( '/'.join(save.split('/')[:-1]) )
    
    mc_sel   = re.compile('(?!(%s))'%('|'.join(data+omit))) if len(data+omit)>0 else re.compile('')
    data_sel = re.compile('('+'|'.join(data)+')(?!.*incl)')
    bkg_sel  = re.compile('(?!(%s))'%('|'.join(data+signals+omit))) if len(data+signals+omit)>0 else re.compile('')

    if histo is None:
        processes = [ p[0] for p in output.values().keys() if not p[0] in data ]
        histogram = output.copy()
    else:
        processes = [ p[0] for p in output[histo].values().keys() if not p[0] in data ]
        histogram = output[histo].copy()

    histogram = histogram.project(axis, 'dataset')
    if overlay: overlay = overlay.project(axis, 'dataset')
    if bins:
        histogram = histogram.rebin(axis, bins)
        if overlay: overlay = overlay.rebin(axis, bins)
    
    y_max = histogram[bkg_sel].sum("dataset").values(overflow='over')[()].max()
    print(histogram[bkg_sel].sum("dataset").values(overflow='over')[()].max())
    MC_total = histogram[bkg_sel].sum("dataset").values(overflow='over')[()].sum()
    Data_total = 0
    if data:
        Data_total = histogram[data_sel].sum("dataset").values(overflow='over')[()].sum()
        #observation = histogram[data[0]].sum('dataset').copy()
        #first = True
        #for d in data:
        #    print (d)
        #    if not first:
        #        observation.add(histogram[d].sum('dataset'))
        #        print ("adding")
        #    first = False
    
    print ("Data:", round(Data_total,0), "MC:", round(MC_total,2))
    
    rescale_tmp = { process: 1 if not process in rescale else rescale[process] for process in processes }
    if normalize and data_sel:
        scales = { process: Data_total*rescale_tmp[process]/MC_total for process in processes }
        histogram.scale(scales, axis='dataset')
    else:
        scales = rescale_tmp

    if shape:
        scales = { process: 1/histogram[process].sum("dataset").values(overflow='over')[()].sum() for process in processes }
        histogram.scale(scales, axis='dataset')
    
    if data:
        fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    else:
        fig, ax  = plt.subplots(1,1,figsize=(10,10) )

    if overlay:
        ax = hist.plot1d(overlay, overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None, binwnorm=binwnorm)

    if shape:
        ax = hist.plot1d(histogram[bkg_sel], overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None, binwnorm=binwnorm)
    else:
        ax = hist.plot1d(histogram[bkg_sel], overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, order=(order if order else processes), binwnorm=binwnorm)

    if signals:
        for sig in signals:
            ax = hist.plot1d(histogram[sig], overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None, binwnorm=binwnorm)
    if data:
        ax = hist.plot1d(histogram[data_sel].sum("dataset"), ax=ax, overflow='over', error_opts=data_err_opts, clear=False, binwnorm=binwnorm)
        #ax = hist.plot1d(observation, ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

        hist.plotratio(
                num=histogram[data_sel].sum("dataset"),
                denom=histogram[bkg_sel].sum("dataset"),
                ax=rax,
                error_opts=data_err_opts,
                denom_fill_opts=None, # triggers this: https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L376
                guide_opts={},
                unc='num',
                #unc=None,
                overflow='over'
        )
    
    
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = []
    for handle, label in zip(handles, labels):
        try:
            if label is None or label=='None':
                updated_labels.append(obs_label)
                handle.set_color('#000000')
            else:
                updated_labels.append(new_labels[label])
                handle.set_color(new_colors[label])
        except:
            pass

    if data:
        if ratio_range:
            rax.set_ylim(*ratio_range)
        else:
            rax.set_ylim(0.1,1.9)
        rax.set_ylabel('Obs./Pred.')
        if axis_label:
            rax.set_xlabel(axis_label)

    ax.set_xlabel(axis_label)
    ax.set_ylabel(y_axis_label)
    
    if not binwnorm:
        if not shape:
            addUncertainties(ax, axis, histogram, bkg_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=False, scales=scales)

        if data:
            addUncertainties(rax, axis, histogram, bkg_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=True, scales=scales)
    
    if log:
        ax.set_yscale('log')
        
    y_mult = 1.7 if not log else 100
    
    if ymax:
        ax.set_ylim(0.01, ymax)
    else:
        y_max = y_max*y_mult*(Data_total/MC_total) if data else y_max*y_mult
        ax.set_ylim(0.01, y_max if not shape else 2)
        #if binwnorm: ax.set_ylim(0.5)

    ax.legend(
        loc='upper right',
        ncol=2,
        borderaxespad=0.0,
        labels=updated_labels,
        handles=handles,
    )
    plt.subplots_adjust(hspace=0)

    hep.cms.label(
        "Preliminary",
        data=len(data)>0 and is_data,
        lumi=lumi,
        loc=0,
        ax=ax,
    )

    if normalize:
        fig.text(0.55, 0.55, 'Data/MC = %s'%round(Data_total/MC_total,2), fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )


    if save:
        #fig.savefig("{}.pdf".format(save))
        fig.savefig("{}.png".format(save))
        print ("Figure saved in:", save)

def get_total(histos, keys):
        tmp = Hist1D.from_bincounts(np.zeros(len(histos[keys[0]].counts)), histos[keys[0]].edges, )
        for key in keys:
            tmp += histos[key]
        return tmp

def add_uncertainty(hist, ax, ratio=False):
    opts = {'step': 'post', 'label': 'Uncertainty', 'hatch': '///',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0, 'zorder':10.}
    
    if ratio:
        down = np.ones(len(hist.counts)) - hist.errors/hist.counts
        up = np.ones(len(hist.counts)) + hist.errors/hist.counts
    else:
        down = hist.counts-hist.errors
        up = hist.counts+hist.errors
    ax.fill_between(x=hist.edges, y1=np.r_[down, down[-1]], y2=np.r_[up, up[-1]], **opts)
    
def addUncertainties(ax, axis, h, selection, up_vars, down_vars, overflow='over', rebin=False, ratio=False, scales={}):
    
    if rebin:
        h = h.project(axis, 'dataset').rebin(axis, rebin)
    
    bins = h[selection].axis(axis).edges(overflow=overflow)
    
    values = h[selection].project(axis, 'dataset').sum('dataset').values(overflow=overflow, sumw2=True)[()]
    central = values[0]
    stats = values[1]
    
    #print (central)
    
    up = np.zeros_like(central)
    down = np.zeros_like(central)
    
    for up_var in up_vars:
        if rebin:
            up_var = up_var.project(axis, 'dataset').rebin(axis, rebin)
            up_var.scale(scales, axis='dataset')
        #print (up_var[selection].values())
        up += (up_var[selection].sum('dataset').values(overflow=overflow, sumw2=False)[()] - central)**2
    
    for down_var in down_vars:
        if rebin:
            down_var = down_var.project(axis, 'dataset').rebin(axis, rebin)
            down_var.scale(scales, axis='dataset')
        down += (down_var[selection].sum('dataset').values(overflow=overflow, sumw2=False)[()] - central)**2
    
    up   += stats 
    down += stats
 
    if ratio:
        up = np.ones_like(central) + np.sqrt(up)/central
        down = np.ones_like(central) - np.sqrt(down)/central
    else:
        up = central + np.sqrt(up)
        down = central - np.sqrt(down)
    
    opts = {'step': 'post', 'label': 'uncertainty', 'hatch': '///',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0, 'zorder':100.}
    
    ax.fill_between(x=bins, y1=np.r_[down, down[-1]], y2=np.r_[up, up[-1]], **opts)
    
def makePlot2(output, histo, axis, bins, xlabel, labels, colors,
              signals=[],
              plot_dir = '/home/users/$USER/public_html/HbbMET/background/',
              shape=False):
        histos = {}
        
        tmp1 = output[histo].copy()
        tmp1 = tmp1.rebin(axis, bins)
        
        keys = tmp1.values().keys()
        
        overbins = 'all'
        if histo in ['n_AK4', 'NH_weight', 'AK8_sdmass']:
            overbins = 'over'
                
        for sample in keys:
            h1 = Hist1D.from_bincounts(
                tmp1.values(overflow = overbins)[sample].T,
                (tmp1.axis(axis).edges(overflow = overbins)),
            )
            histos[sample] = h1

        edge = list(keys)[0]
        
        backgrounds = []
        for sample in keys:
            if sample not in signals:
                backgrounds += (sample,)
                           
        order = [('other',), ('WJetsToLNu',), ('TT',), ('QCD_bEnriched_HT',), ('ZJetsToNuNu_HT',),]
        total_mc = get_total(histos, backgrounds)
        
        fig, (ax) = plt.subplots(figsize=(11.75,10))
        hep.cms.label(
            'Phase-2 Simulation Preliminary',
            data=True,
            loc=0,
            ax=ax,
            #lumi = 3000,
            rlabel = r'$3000\ fb^{-1}\ (14\ TeV)$',
        )
        
        if shape:
            back_hist_type = 'step'
            back_stack = False
            density = True
            y_label = 'a.u.'
            
        if not shape:
            back_hist_type = 'fill'
            back_stack = True
            density = False    
            y_label = r'Events'
        
        if backgrounds != []:
            hep.histplot(
                [histos[sample].counts for sample in order],
                histos[edge].edges,
                #w2=[(hists[x].errors)**2 for x in keys ],
                histtype= back_hist_type,
                density = density,
                stack = back_stack,
                label = [labels[sample] for sample in order],
                color = [colors[sample] for sample in order],
                ax = ax
            )
        
        hep.histplot(
            [histos[sample].counts for sample in signals[1:5]],
            histos[edge].edges,
            #w2=[(histos[sample].errors)**2 for sample in signals[1:4]],
            histtype = "step",
            density = density,
            stack = False,
            label = [labels[sample] for sample in signals[1:5]],
            color = list(colors.values())[0:4],
            #linestyle = '-',
            ax = ax
        )
        
        #plt.axvline(50, linestyle='--', color='gray')
        #plt.axvline(100, linestyle='--', color='gray')
        #plt.axvline(150, linestyle='--', color='gray')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.set_yscale('log')
        ax.legend(prop={'size': 11})
        if histo in ['met_pt']:
            ax.set_xlim(100)
        if histo in ['AK4_QCD_veto', 'AK8_QCD_veto', 'dphi_AK4_MET', 'dphi_AK8_MET']:
            ax.set_xlim(right=3.5)
        if histo in ['AK8_sdmass', 'met_pt', 'min_AK8_pt', 'n_AK4']:
            ax.set_ylim(1e-5)
        
        #add_uncertainty(total_mc, ax, ratio=True)

        plot_dir = os.path.expandvars(plot_dir)
        finalizePlotDir(plot_dir)
        fig.savefig(plot_dir+str(histo)+'.png')
        fig.savefig(plot_dir+str(histo)+'.pdf')

def scale_and_merge_histos(histogram, samples, fileset, lumi=3000):
    """
    Scale samples to a physical cross section.
    Merge samples into categories, e.g. several ttZ samples into one ttZ category.
    
    histogram -- coffea histogram
    samples -- samples dictionary that contains the x-sec and sumWeight
    fileset -- fileset dictionary used in the coffea processor
    lumi -- integrated luminosity in 1/fb
    """
    temp = histogram.copy()

    # scale according to cross sections    
    scales = {sample: lumi*1000*samples[sample]['xsec']/samples[sample]['nevents'] for sample in samples if sample in fileset}
    temp.scale(scales, axis='dataset')

    
    # merge according to categories:
    mapping = {
        'ZJetsToNuNu_HT': [
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT2500toInf_HLLHC',
        ],
        'WJetsToLNu': [
            'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        ],
        'QCD_bEnriched_HT': [
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        ],
        'other': [
            'ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU',
            'WminusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU',
            'WplusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU',
        ],
        'TT': [
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',
            'TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU',
            'ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU',
            'ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU',
            'ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
            'ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
        ],
         '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000'
        ],
        '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250'
        ],
    }
    temp = temp.group("dataset", hist.Cat("dataset", "new grouped dataset"), mapping)
                
    return temp

def compute_darkness(r, g, b, a=1.0):
    """Compute the 'darkness' value from RGBA (darkness = 1 - luminance)
       stolen from Nick Amin: https://github.com/aminnj/yahist
       Version from Jonathan Guiang: https://gist.github.com/jkguiang/279cb4d2e68e64148afc62274df09f18
    """
    return a * (1.0 - (0.299 * r + 0.587 * g + 0.114 * b))

def bin_text(counts, x_edges, y_edges, axes, cbar, errors=None, size=10, fmt=":0.2e"):
    """Write bin population on top of 2D histogram bins,
       stolen from Nick Amin: https://github.com/aminnj/yahist
       Version from Jonathan Guiang: https://gist.github.com/jkguiang/279cb4d2e68e64148afc62274df09f18
    """
    show_errors = (type(errors) != type(None))
    x_centers = x_edges[1:]-(x_edges[1:]-x_edges[:-1])/2
    y_centers = y_edges[1:]-(y_edges[1:]-y_edges[:-1])/2
    
    if show_errors:
        label_template = r"{0"+fmt+"}\n$\pm{1:0.2f}\%$"
    else:
        errors = np.zeros(counts.shape)
        label_template = r"{0"+fmt+"}"
        
    xyz = np.c_[        
        np.tile(x_centers, len(y_centers)),
        np.repeat(y_centers, len(x_centers)),
        counts.flatten(),
        errors.flatten()
    ][counts.flatten() != 0]

    r, g, b, a = cbar.mappable.to_rgba(xyz[:, 2]).T
    colors = np.zeros((len(xyz), 3))
    colors[compute_darkness(r, g, b, a) > 0.45] = 1

    for (x, y, count, err), color in zip(xyz, colors):
        axes.text(
            x,
            y,
            label_template.format(count, err),
            color=color,
            ha="center",
            va="center",
            fontsize=size,
            wrap=True,
        )

    return
