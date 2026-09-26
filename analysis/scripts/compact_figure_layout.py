"""Compact layout fitted to the complete rendered footprint of each panel."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgb,LinearSegmentedColormap
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,save_figure_with_subfigures
from scientific_palettes import get_palette

configure_publication_style()
plt.rcParams.update({'axes.labelsize':15.5,'xtick.labelsize':12.5,'ytick.labelsize':12.5,'legend.fontsize':11.5,
 'axes.labelpad':4,'xtick.major.pad':3,'ytick.major.pad':3,'lines.linewidth':1.8,
 'font.weight':'normal','axes.labelweight':'normal','axes.titleweight':'normal','figure.titleweight':'normal'})
PAL=get_palette('nature_new',8); INK='#28353B'
BLUE=PAL[1];TEAL=PAL[3];CORAL=PAL[0];PURPLE=PAL[4];GREEN=PAL[2];GOLD=PAL[6];LILAC=PAL[7]
plt.rcParams.update({'text.color':INK,'axes.labelcolor':INK,'axes.unicode_minus':True})
def line_color(color):
 """Keep the exact categorical palette hue, including for thin series lines."""
 assert color in PAL,color
 return color

def palette_sequential(color):
 """Continuous numeric colour scale: white to an exact nature_new endpoint."""
 assert color in PAL,color
 cmap=LinearSegmentedColormap.from_list('nature_new_sequential',['white',color])
 cmap.set_bad('white')
 return cmap

def axis(ax,x=None,y=None,ticks=12.5):
 style_axis(ax,x,y,tick_size=ticks,label_size=15.5)
 for label in [ax.xaxis.label,ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels():label.set_fontweight('normal')
 ax.xaxis.labelpad=5;ax.yaxis.labelpad=5
 for label in ax.get_xticklabels():label.set_rotation(0);label.set_ha('center')

def legend(ax,ncol=2,fontsize=11.5,handlelength=1.35,handles=None,columnspacing=.9):
 handles,labels=(ax.get_legend_handles_labels() if handles is None else (handles,[h.get_label() for h in handles]))
 # Matplotlib fills columns; reorder so readers scan the scientific order by row.
 order=[i for col in range(ncol) for i in range(col,len(handles),ncol)]
 return ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower left',bbox_to_anchor=(0,1.025),ncol=ncol,fontsize=fontsize,
  borderaxespad=0,borderpad=0,handlelength=handlelength,handletextpad=.45,columnspacing=columnspacing,labelspacing=.35)

def colorbar(ax,im,label,ticks):
 cbax=ax.inset_axes([0,1.15,1,.045]);cb=ax.figure.colorbar(im,cax=cbax,orientation='horizontal',ticks=ticks)
 cb.set_label(label,fontsize=12,labelpad=4);cb.ax.xaxis.set_label_position('top')
 cb.ax.minorticks_off();cb.ax.tick_params(labelsize=11,length=4,width=.8,pad=3,colors=INK)
 cb.outline.set_linewidth(.7);cb.outline.set_edgecolor(INK)
 return cb

def cell_ink(rgba):
 rgb=np.asarray(rgba[:3]);ink=np.asarray(to_rgb(INK))
 def luminance(c):return float(np.dot(np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4),[.2126,.7152,.0722]))
 lum=luminance(rgb);dark=luminance(ink)
 return 'white' if 1.05/(lum+.05)>(lum+.05)/(dark+.05) else INK

def matrix_dividers(ax,rows,cols):
 # Fine cell boundaries improve row tracking without adding plot grids.
 ax.hlines(np.arange(rows-1)+.5,-.5,cols-.5,color='white',lw=.5,alpha=.55)
 ax.vlines(np.arange(cols-1)+.5,-.5,rows-.5,color='white',lw=.5,alpha=.55)

def detail_audit(fig,axes):
 renderer=fig.canvas.get_renderer();out=[]
 for ax in axes:
  checks=[]
  for dim in [ax.xaxis,ax.yaxis]:
   low,high=sorted(dim.get_view_interval());locs=dim.get_majorticklocs()
   labels=[t for v,t in zip(locs,dim.get_ticklabels()) if low-1e-10<=v<=high+1e-10 and t.get_visible() and t.get_text()]
   boxes=[t.get_window_extent(renderer) for t in labels]
   checks.extend(a.overlaps(b) for a,b in zip(boxes,boxes[1:]))
  leg=ax.get_legend();clearance=None
  if leg is not None:
   clearance=(leg.get_window_extent(renderer).y0-ax.get_window_extent(renderer).y1)*72/fig.dpi
  out.append({'adjacent_major_tick_overlap':any(checks),'legend_clearance_above_axes_pt':clearance})
 return out

def footprint(fig,ax):
 fig.canvas.draw()
 return ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())

def fit_to_cells(fig,axes,targets):
 """Fit visible labels/legend/colourbar bounds, rather than only axis spines."""
 w,h=fig.get_size_inches()
 for iteration in range(55):
  for text in fig.findobj(match=Text):text.set_fontweight('normal')
  fig.canvas.draw();renderer=fig.canvas.get_renderer();worst=0
  for ax,target in zip(axes,targets):
   bb=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
   delta=np.asarray(target.extents)-np.asarray(bb.extents)
   worst=max(worst,float(np.max(np.abs(delta))))
   p=ax.get_position();new=np.array([p.x0*w,p.y0*h,p.x1*w,p.y1*h])+.8*delta
   if new[2]-new[0]<1.05 or new[3]-new[1]<.7:
    raise ValueError(f'Labels exceed panel capacity: {ax.get_label()}, {new}')
   ax.set_position([new[0]/w,new[1]/h,(new[2]-new[0])/w,(new[3]-new[1])/h])
  if worst<.004:break
 boxes=[footprint(fig,a).extents.tolist() for a in axes]
 errors=[float(np.max(np.abs(np.asarray(b)-t.extents))) for b,t in zip(boxes,targets)]
 if max(errors)>.025:raise ValueError(f'Visual-boundary fitting did not converge: {errors}')
 return {'iterations':iteration+1,'max_boundary_error_pt':max(errors)*72,'outer_bounds_inches':boxes,'target_bounds_inches':[t.extents.tolist() for t in targets]}

def build_canvas(builders,rows,cols,size,gap=(.4,.36),row_ratios=None):
 fig=plt.figure(figsize=size)
 w,h=size;mx=.16;my=.14
 cw=(w-2*mx-(cols-1)*gap[0])/cols
 ratios=np.ones(rows) if row_ratios is None else np.asarray(row_ratios,float)
 assert len(ratios)==rows and np.all(ratios>0)
 heights=(h-2*my-(rows-1)*gap[1])*ratios/ratios.sum()
 axes=[];targets=[]
 for k,builder in enumerate(builders):
  row,col=divmod(k,cols);ch=heights[row];left=mx+col*(cw+gap[0]);bottom=h-my-heights[:row+1].sum()-row*gap[1]
  target=Bbox.from_bounds(left,bottom,cw,ch)
  ax=fig.add_axes([(left+.65)/w,(bottom+.6)/h,(cw-.8)/w,(ch-1.0)/h],label=f'panel_{k}')
  builder(ax)
  # Fixed categorical tick labels must be changed through the axis formatter.
  # Editing transient Text artists alone is undone on the next canvas draw.
  for dimension,setter in [(ax.xaxis,ax.set_xticks),(ax.yaxis,ax.set_yticks)]:
   labels=[t.get_text() for t in dimension.get_ticklabels()]
   renamed=[s.replace('History','Pair mean').replace('Block:','Condition:') for s in labels]
   if renamed!=labels:
    bounds=ax.get_xlim(),ax.get_ylim();setter(dimension.get_ticklocs(),renamed)
    ax.set_xlim(bounds[0]);ax.set_ylim(bounds[1])
  if ax.get_xlabel()=='Paired mean\nsquared error':ax.set_xlabel('Paired mean squared\nerror (pp²)')
  axes.append(ax);targets.append(target)
 for text in fig.findobj(match=Text):text.set_fontweight('normal')
 report=fit_to_cells(fig,axes,targets)
 report.update({'rows':rows,'columns':cols,'canvas_inches':list(size),'visual_gap_inches':list(gap),'tick_rotation_degrees':[[float(t.get_rotation()) for t in ax.get_xticklabels()] for ax in axes],
  'text_weights':sorted(set(str(t.get_fontweight()) for t in fig.findobj(match=Text) if t.get_text())),
  'detail_checks':detail_audit(fig,axes)})
 return fig,axes,report

PANEL_LABEL_PT = 24
PANEL_LABEL_OFFSETS_PT = {('Fig05_TOPSIS_robustness', 'f', False): [-12, 0], ('Fig05_TOPSIS_robustness', 'f', True): [56, 0], ('Fig06_material_comparisons', 'a', False): [0, -3], ('Fig06_material_comparisons', 'a', True): [0, -3.5], ('Fig07_adsorption_responses', 'a', False): [0, -3], ('Fig07_adsorption_responses', 'b', False): [0, -3], ('Fig07_adsorption_responses', 'h', False): [0, -2]}

def panel_letter(fig,bounds,size,stem,letter,standalone=False):
 """Regular 24 pt panel letters, with reviewed clearance from labels/legends."""
 dx,dy=PANEL_LABEL_OFFSETS_PT.get((stem,letter,standalone),(0,0))
 return fig.text((bounds[0]+dx/72)/size[0],(bounds[3]-dy/72)/size[1],letter,
  ha='left',va='top',fontsize=PANEL_LABEL_PT,fontweight='normal')


def render_figure(folder,stem,builders,rows,cols,size,tables,row_ratios=None,standalone_sizes=None,panel_letters=False):
 folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
 for name,table in tables.items():table.to_csv(folder/f'{name}.csv',index=True,encoding='utf-8-sig')
 fig,axes,report=build_canvas(builders,rows,cols,size,row_ratios=row_ratios)
 if panel_letters:
  for i,bounds in enumerate(report['target_bounds_inches']):
   panel_letter(fig,bounds,size,stem,chr(97+i))
  report['panel_letters']=[chr(97+i) for i in range(len(builders))]
  report.update(panel_label_pt=PANEL_LABEL_PT,panel_label_weight='normal')
 reports=[]
 def export_independent():
  sub=folder/'subfigures';sub.mkdir(exist_ok=True)
  lines=['# Subfigures','','Each panel is independently redrawn from its data builder on a standalone canvas. These files are not crops of the combined figure.','']
  for i,builder in enumerate(builders):
   # Use a fresh figure and axes; rerun all artists including legends/colourbars.
   canvas=(standalone_sizes or {}).get(i,(5.5,4.8 if len(builders)==4 else 4.3))
   sf,sa,sr=build_canvas([builder],1,1,canvas,gap=(0,0))
   if panel_letters:
    bb=sr['target_bounds_inches'][0]
    panel_letter(sf,bb,canvas,stem,chr(97+i),standalone=True)
    sr.update(panel_label_pt=PANEL_LABEL_PT,panel_label_weight='normal')
   letter=chr(97+i);target=sub/f'{stem}_{letter}'
   sf.savefig(target.with_suffix('.png'),dpi=600,facecolor='white')
   sf.savefig(target.with_suffix('.pdf'),facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
   qa=folder.parent/'qa'/'standalone_previews';qa.mkdir(parents=True,exist_ok=True)
   sf.savefig(qa/f'{stem}_{letter}.png',dpi=130,facecolor='white')
   plt.close(sf);reports.append(sr)
   lines.append(f'- {letter}: [{target.name}.png](<{target.with_suffix(".png").as_posix()}>) · [{target.name}.pdf](<{target.with_suffix(".pdf").as_posix()}>)')
  (sub/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  return sub
 fig._standalone_exporter=export_independent
 fig.savefig(folder/f'{stem}_preview.png',dpi=160,facecolor='white')
 save_figure_with_subfigures(fig,folder,stem,dpi=600)
 report['standalone_panels']=reports;report['palette']='nature_new';report['palette_base_colors']=PAL;report['axis_label_pt']=15.5;report['tick_label_pt']=12.5
 (folder/'layout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('SAVED',stem,'panels',len(builders),'boundary_error_pt',round(report['max_boundary_error_pt'],3),flush=True)
