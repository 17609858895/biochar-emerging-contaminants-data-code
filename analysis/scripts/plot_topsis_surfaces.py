"""Fig. 5: six retained diagnostics and three dense preference-map panels."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import compact_figure_layout as lay
import plot_full24_revision as base
from plot_panel_integration import capture

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/topsis_surfaces_v1'
OUT=ROOT/'figures/revision_full24_20260912/Fig05';STEM='Fig05_TOPSIS_robustness'
INK=lay.INK;PAL=lay.PAL
CM=LinearSegmentedColormap.from_list('nature_new_blue_coral',[PAL[1],PAL[0]])
grid=np.load(E/'grid_and_selections.npz');XY=grid['vertices'];TRI=grid['triangles'];CENTRES=grid['centres']
G=pd.read_csv(E/'broad_triangles.csv');EQ=pd.read_csv(E/'broad_equal_weight.csv').iloc[0]
EDGES=np.load(E/'broad_boundary_edges.npy')
original=capture(base.fig5)
adj={}
for i,t in enumerate(TRI):
 for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:adj.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)

def old_builder(builder):
 def draw(ax):
  builder(ax)
  for txt in ax.findobj(Text):txt.set_fontsize(max(13.5,txt.get_fontsize()*1.24));txt.set_fontweight('normal')
  if ax.get_ylabel()=='Nondominated inner methods':ax.set_ylabel('Nondominated\ninner methods',fontsize=18)
  if ax.get_legend()is not None:
   ax.get_legend().set_loc('lower right');ax.get_legend().set_bbox_to_anchor((1,1.025))
  for collection in ax.collections:
   if hasattr(collection,'get_sizes')and len(collection.get_sizes()):collection.set_sizes(collection.get_sizes()*1.35)
 return draw

def surface(metric,label,factor):
 def draw(ax):
  z=G[metric+'_median'].to_numpy()*factor;norm=Normalize(z.min(),z.max())
  xyz=np.dstack([XY[TRI],np.repeat(z[:,None,None],3,axis=1)])
  coll=Poly3DCollection(xyz,facecolors=CM(norm(z)),edgecolors='none',linewidths=0,antialiased=False,zsort='average')
  ax.add_collection3d(coll)
  # Vertical facets retain discontinuities; no sloping interpolation across models.
  walls=[];wallcolors=[]
  for (a,b),ids in adj.items():
   if len(ids)!=2:continue
   za,zb=z[ids]
   if abs(za-zb)<1e-12:continue
   walls.append([[*XY[a],za],[*XY[b],za],[*XY[b],zb],[*XY[a],zb]])
   wallcolors.append(CM(norm((za+zb)/2)))
  ax.add_collection3d(Poly3DCollection(walls,facecolors=wallcolors,edgecolors=INK,linewidths=.10,alpha=1))
  ax.scatter([1/3],[1/3],[EQ[metric+'_median']*factor],marker='*',s=115,c=PAL[3],edgecolors=INK,linewidths=.7,depthshade=False,zorder=20)
  ax.set_xlim(.08,.82);ax.set_ylim(.08,.82);span=np.ptp(z);ax.set_zlim(z.min()-.06*span,z.max()+.04*span)
  ax.set_xticks([.1,.4,.7]);ax.set_yticks([.1,.4,.7]);ax.zaxis.set_major_locator(MaxNLocator(3))
  ax.set_xlabel(r'$w_{\mathrm{point}}$',fontsize=17,labelpad=2)
  ax.set_ylabel(r'$w_{\mathrm{contrast}}$',fontsize=17,labelpad=2)
  ax.set_zlabel(label,fontsize=17,labelpad=4)
  for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_rotate_label(True)
  ax.view_init(29,-128);ax.set_box_aspect((1,1,.72));ax.tick_params(labelsize=13.5,pad=0)
  for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
   axis.set_pane_color((1,1,1,0));axis._axinfo['grid']['linewidth']=0
  draw.last_range=[float(z.min()),float(z.max())]
 draw.projection='3d'
 return draw

def retention_map(ax):
 vals=G.winner_retention_pct.to_numpy();cm=LinearSegmentedColormap.from_list('nature_new_retention',['white',PAL[3]])
 im=ax.tripcolor(Triangulation(XY[:,0],XY[:,1],TRI),facecolors=vals,cmap=cm,vmin=0,vmax=100,shading='flat',edgecolors='none',rasterized=True)
 ax.add_collection(LineCollection(XY[EDGES],colors=INK,linewidths=.32,alpha=.65))
 ax.plot([.1,.8,.1,.1],[.1,.1,.8,.1],c=INK,lw=.8)
 ax.scatter([1/3],[1/3],marker='*',s=170,c=PAL[3],edgecolors=INK,lw=.8,zorder=5)
 ax.scatter([.65],[.74],marker='*',s=110,c=PAL[3],edgecolors=INK,lw=.7)
 ax.text(.69,.74,'Equal\nweights',ha='left',va='center',fontsize=13.5)
 lay.axis(ax,r'$w_{point}$',r'$w_{contrast}$',ticks=13.5)
 ax.xaxis.label.set_fontsize(18);ax.yaxis.label.set_fontsize(18)
 ax.set_xlim(.08,.87);ax.set_ylim(.08,.85);ax.set_xticks([.1,.4,.7]);ax.set_yticks([.1,.4,.7])
 cb=lay.colorbar(ax,im,'Retained selections (%)',[0,50,100]);cb.ax.xaxis.label.set_fontsize(14);cb.ax.tick_params(labelsize=13)

BUILDERS=[old_builder(b)for b in original['builders']]+[surface('contrast_mae','Contrast MAE (pp)',100),surface('selection_loss',r'Selection loss ($10^{-2}$ pp)',10000),retention_map]

def full3d_bbox(fig,ax):
 renderer=fig.canvas.get_renderer()
 return Bbox.union([ax.get_tightbbox(renderer)]+[axis.label.get_window_extent(renderer)for axis in [ax.xaxis,ax.yaxis,ax.zaxis]if axis.label.get_text()]).transformed(fig.dpi_scale_trans.inverted())

def fit3d(fig,ax,target):
 # Fit the complete projected axes plus labels isotropically inside the panel.
 fw,fh=fig.get_size_inches()
 for k in range(15):
  fig.canvas.draw();bb=full3d_bbox(fig,ax)
  fac=min(target.width/bb.width,target.height/bb.height)
  p=ax.get_position();px,py,pw,ph=p.x0*fw,p.y0*fh,p.width*fw,p.height*fh
  if abs(fac-1)<.004:break
  ax.set_position([px/fw,py/fh,pw*fac/fw,ph*fac/fh])
 fig.canvas.draw();bb=full3d_bbox(fig,ax)
 p=ax.get_position();dx=target.x0+(target.width-bb.width)/2-bb.x0;dy=target.y0+(target.height-bb.height)/2-bb.y0
 ax.set_position([p.x0+dx/fw,p.y0+dy/fh,p.width,p.height]);fig.canvas.draw()
 bb=full3d_bbox(fig,ax)
 assert bb.x0>=target.x0-.025 and bb.x1<=target.x1+.025 and bb.y0>=target.y0-.025 and bb.y1<=target.y1+.025,(bb,target)
 return bb.extents.tolist()

def canvas(indices,size,rows,cols):
 fig=plt.figure(figsize=size);fw,fh=size;mx=.18;my=.16;gx=.36;gy=.30
 cw=(fw-2*mx-gx*(cols-1))/cols;ch=(fh-2*my-gy*(rows-1))/rows
 axlist=[];targets=[];tags=[];twod=[];tdtargets=[]
 for pos,k in enumerate(indices):
  row,col=divmod(pos,cols);left=mx+col*(cw+gx);bottom=fh-my-(row+1)*ch-row*gy
  # A dedicated regular 24pt tag band is kept above each panel's full footprint.
  target=Bbox.from_bounds(left,bottom,cw,ch-.36)
  is3d=getattr(BUILDERS[k],'projection',None)=='3d'
  ax=fig.add_axes([(left+.75)/fw,(bottom+.65)/fh,(cw-1.0)/fw,(ch-1.45)/fh],**({'projection':'3d','computed_zorder':False} if is3d else {}))
  BUILDERS[k](ax);axlist.append(ax);targets.append(target)
  if not is3d:twod.append(ax);tdtargets.append(target)
  tag=fig.text(left/fw,(bottom+ch)/fh,chr(97+k),fontsize=24,weight='normal',va='top',ha='left');tags.append(tag)
 for t in fig.findobj(Text):t.set_fontweight('normal')
 report=lay.fit_to_cells(fig,twod,tdtargets) if twod else {}
 projected=[]
 for k,ax,target in zip(indices,axlist,targets):
  if getattr(BUILDERS[k],'projection',None)=='3d':projected.append(fit3d(fig,ax,target))
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 boxes=[full3d_bbox(fig,a)if getattr(BUILDERS[k],'projection',None)=='3d'else a.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())for k,a in zip(indices,axlist)]
 for i,a in enumerate(boxes):
  for b in boxes[i+1:]:assert not a.overlaps(b),'Panel footprint overlap'
 for tag,a in zip(tags,boxes):assert not tag.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()).overlaps(a),'Panel letter overlap'
 report.update(canvas_inches=list(size),rows=rows,cols=cols,projected_3d_bounds_inches=projected,all_panel_bounds_inches=[b.extents.tolist()for b in boxes],panel_letters=[chr(97+k)for k in indices],panel_label_pt=24,panel_label_weight='normal',palette='nature_new',palette_base_colors=PAL,detail_checks=lay.detail_audit(fig,[a for k,a in zip(indices,axlist)if getattr(BUILDERS[k],'projection',None)!='3d']),surface_rendering='flat centroid triangles with vertical jump walls',no_interpanel_overlap=True)
 return fig,report

def main():
 OUT.mkdir(exist_ok=True,parents=True);sub=OUT/'subfigures';sub.mkdir(exist_ok=True)
 for name,table in original['tables'].items():table.to_csv(OUT/(name+'.csv'),index=True,encoding='utf-8-sig')
 for n in ['broad_vertices.csv','broad_triangles.csv','broad_equal_weight.csv']:
  (OUT/n).write_bytes((E/n).read_bytes())
 fig,report=canvas(list(range(9)),(15.0,14.0),3,3);subreports=[]
 def export_independent():
  for k in range(9):
   sf,sr=canvas([k],(5.5,4.8),1,1);target=sub/(STEM+'_'+chr(97+k))
   sf.savefig(target.with_suffix('.png'),dpi=600,facecolor='white');sf.savefig(target.with_suffix('.pdf'),facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
   sf.savefig(OUT/(f'qa_{chr(97+k)}.png'),dpi=130,facecolor='white');plt.close(sf);subreports.append(sr)
  (sub/'README.md').write_text('# Independent panels\n\nAll nine panels are independently redrawn from their data builders. PNG 600 dpi and PDF.\n',encoding='utf8')
  return sub
 fig._standalone_exporter=export_independent
 fig.savefig(OUT/(STEM+'_preview.png'),dpi=150,facecolor='white')
 lay.save_figure_with_subfigures(fig,OUT,STEM,dpi=600)
 report['standalone_panels']=subreports
 (OUT/'layout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print('SAVED',OUT,flush=True)
if __name__=='__main__':main()
