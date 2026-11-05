/* s4lab_shim.c -- koumyou_sim/Iridis 用の、S4コア（third_party/S4/S4/、本家 victorliu/S4）への
 * 薄いCラッパー。RCWA2D/s4lab.py が ctypes 経由で呼ぶ。
 *
 * 本家S4のPythonバインディング(S4/main_python.c)はコアのAPI刷新（S4_MaterialID/S4_LayerIDの
 * 整数ID化）に追従しておらずビルドできない（Python2専用API・型不一致に加え、kwlistの
 * "Material"/"Layer"が"S4_Material"/"S4_Layer"に一括置換されている）。そのため
 * main_python.c は使わず、S4.h の公開API（＋S4_internal.h の一部）を直接呼ぶ。
 *
 * 平面波励振は、旧Pythonバインディング（phoebe-p/S4 フォークの SetExcitationPlanewave）と
 * 同じ Simulation_MakeExcitationPlanewave() を呼ぶので、角度・s/p振幅の規約は従来と同一。
 * 戻り値は全関数とも 0 = 成功、それ以外 = S4 のエラーコード（負値は引数エラー）。
 */
#include <math.h>
#include <string.h>
#include "S4.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* S4_internal.h では C++ のときだけ宣言される（定義は extern "C" なので C から呼べる） */
void Simulation_InvalidateFieldCache(S4_Simulation *S);

/* Lr = {ux, uy, vx, vy}。1D格子は vx=vy=0。nG は要求する基底数（S4が丸める）。 */
S4_Simulation* s4l_new(const double *Lr, int nG){
	return S4_Simulation_New(Lr, (unsigned int)nG, NULL);
}

void s4l_destroy(S4_Simulation *S){
	if(NULL != S){ S4_Simulation_Destroy(S); }
}

/* 実際に使われる基底数を返す。G != NULL なら 2*n 個の整数 (g1,g2) を書き込む。 */
int s4l_get_basis(const S4_Simulation *S, int *G){
	return S4_Simulation_GetBases(S, G);
}

/* 基底（逆格子ベクトルの整数座標 (g1,g2) を n 組）を明示的に設定する。 */
int s4l_set_basis(S4_Simulation *S, int n, int *G){
	return S4_Simulation_SetBases(S, (unsigned int)n, G);
}

/* 名前付き材質のスカラー複素誘電率を設定（無ければ追加）。既存材質の誘電率を変えた場合は
 * キャッシュ済みの層モードを破棄する（S4_Simulation_SetMaterial 自体は破棄しないため）。 */
int s4l_set_material(S4_Simulation *S, const char *name, double eps_re, double eps_im){
	const S4_real eps[2] = { eps_re, eps_im };
	S4_MaterialID M = -1;
	for(int i = 0; i < S->n_materials; ++i){
		if(0 == strcmp(S->material[i].name, name)){ M = i; break; }
	}
	S4_MaterialID ret = S4_Simulation_SetMaterial(S, M, name, S4_MATERIAL_TYPE_SCALAR_COMPLEX, eps);
	if(ret < 0){ return -1; }
	if(M >= 0){
		S4_Simulation_DestroyLayerModes(S, -1);
		Simulation_DestroySolution(S);
	}
	return 0;
}

/* 層を末尾に追加する（最初に追加した層が入射側、最後が透過側）。thickness は um。 */
int s4l_add_layer(S4_Simulation *S, const char *name, double thickness, const char *material){
	S4_MaterialID M = S4_Simulation_GetMaterialByName(S, material);
	if(M < 0){ return 15; }
	const S4_real t = thickness;
	S4_LayerID L = S4_Simulation_SetLayer(S, -1, name, &t, -1, M);
	return (L < 0) ? -1 : 0;
}

static int lookup_layer_material(S4_Simulation *S, const char *layer, const char *material,
                                 S4_LayerID *L, S4_MaterialID *M){
	*L = S4_Simulation_GetLayerByName(S, layer);
	if(*L < 0){ return 13; }
	*M = S4_Simulation_GetMaterialByName(S, material);
	if(*M < 0){ return 15; }
	return 0;
}

static int is_1d_lattice(const S4_Simulation *S){
	return (0 == S->Lr[2] && 0 == S->Lr[3]);
}

/* 矩形領域。angle_deg は度。1D格子では区間(interval)として登録する（第2半幅は無視される）。 */
int s4l_set_region_rectangle(S4_Simulation *S, const char *layer, const char *material,
                             double cx, double cy, double angle_deg, double hwx, double hwy){
	S4_LayerID L; S4_MaterialID M;
	int ret = lookup_layer_material(S, layer, material, &L, &M);
	if(0 != ret){ return ret; }
	const S4_real center[2] = { cx, cy };
	const S4_real angle_frac = angle_deg / 360.0;
	if(is_1d_lattice(S)){
		const S4_real hw[2] = { hwx, 0.0 };
		return S4_Layer_SetRegionHalfwidths(S, L, M, S4_REGION_TYPE_INTERVAL, hw, center, &angle_frac);
	}
	const S4_real hw[2] = { hwx, hwy };
	return S4_Layer_SetRegionHalfwidths(S, L, M, S4_REGION_TYPE_RECTANGLE, hw, center, &angle_frac);
}

/* 円形領域（2D格子のみ）。 */
int s4l_set_region_circle(S4_Simulation *S, const char *layer, const char *material,
                          double cx, double cy, double radius){
	S4_LayerID L; S4_MaterialID M;
	int ret = lookup_layer_material(S, layer, material, &L, &M);
	if(0 != ret){ return ret; }
	const S4_real center[2] = { cx, cy };
	const S4_real hw[2] = { radius, radius };
	const S4_real angle_frac = 0.0;
	return S4_Layer_SetRegionHalfwidths(S, L, M, S4_REGION_TYPE_CIRCLE, hw, center, &angle_frac);
}

/* 多角形領域（2D格子のみ）。v は (x,y) の組を nv 個、反時計回りで渡すこと。
 * 公開APIの S4_Layer_SetRegionVertices は向きの判定に頂点数ではなく層内の図形数 n を使う
 * バグがあり、正しい向きの多角形を反転しうるため、判定のない内部関数を直接呼ぶ。 */
int s4l_set_region_polygon(S4_Simulation *S, const char *layer, const char *material,
                           double cx, double cy, int nv, const double *v){
	int idx;
	S4_Layer *L = Simulation_GetLayerByName(S, layer, &idx);
	if(NULL == L){ return 13; }
	S4_MaterialID M = S4_Simulation_GetMaterialByName(S, material);
	if(M < 0){ return 15; }
	const double center[2] = { cx, cy };
	return Simulation_AddLayerPatternPolygon(S, L, M, center, 0.0, nv, v);
}

/* 周波数 = 1/波長（長さ単位は格子と同じ）。角周波数ではない。 */
int s4l_set_frequency(S4_Simulation *S, double f_re, double f_im){
	const S4_real f[2] = { f_re, f_im };
	return S4_Simulation_SetFrequency(S, f);
}

/* 平面波励振。theta=極角, phi=方位角（度）。s/p振幅は複素数。
 * 入射方向(k_parallel)が変わった場合は層モードを破棄する（Simulation_MakeExcitationPlanewave
 * は S->k を書き換えるだけで層モードのキャッシュを破棄しないため）。 */
int s4l_set_excitation_planewave(S4_Simulation *S, double theta_deg, double phi_deg,
                                 double s_re, double s_im, double p_re, double p_im){
	const double angle[2] = { theta_deg * (M_PI/180.), phi_deg * (M_PI/180.) };
	const double pol_s[2] = { hypot(s_re, s_im), atan2(s_im, s_re) };
	const double pol_p[2] = { hypot(p_re, p_im), atan2(p_im, p_re) };
	const double k_old[2] = { S->k[0], S->k[1] };
	int ret = Simulation_MakeExcitationPlanewave(S, angle, pol_s, pol_p, 0);
	if(0 != ret){ return ret; }
	if(S->k[0] != k_old[0] || S->k[1] != k_old[1]){
		S4_Simulation_DestroyLayerModes(S, -1);
		Simulation_DestroySolution(S);
	}
	return 0;
}

/* 次数ごとのz方向ポインティング束。power は 4*n_G 個:
 * [4i+0]=forward実部, [4i+1]=backward実部, [4i+2]=forward虚部, [4i+3]=backward虚部。
 * 行 i は s4l_get_basis() が返す G の i 番目に対応する。 */
int s4l_get_power_flux_by_order(S4_Simulation *S, const char *layer, double zoffset, double *power){
	int idx;
	S4_Layer *L = Simulation_GetLayerByName(S, layer, &idx);
	if(NULL == L){ return 13; }
	/* 既定の一括ソルバー SolveAll()（rcwa.cpp）は層数2（入射側＋透過側のみ）で右辺の
	 * 格納位置が重なって壊れ、R=T=0 を返す。層数2以下では従来の層ごとのソルバー
	 * SolveInterior() を使わせる（options.use_less_memory で切り替わる）。 */
	if(S->n_layers <= 2 && !S->options.use_less_memory){
		S->options.use_less_memory = 1;
		Simulation_DestroySolution(S);
	}
	return Simulation_GetPoyntingFluxByG(S, L, zoffset, power);
}

/* PolarizationDecomposition（Liの正しいフーリエ因子分解、法線ベクトル場による偏光分解）。 */
int s4l_set_polarization_decomposition(S4_Simulation *S, int on){
	S->options.use_polarization_basis = on ? 1 : 0;
	S4_Simulation_DestroyLayerModes(S, -1);
	Simulation_DestroySolution(S);
	return 0;
}

/* 偏光分解まわりの詳細オプション。負の値の引数は変更しない。
 * resolution  : 法線ベクトル場を作る格子の細かさ（G の広がりに対する倍率、既定 8、2 以上）
 * basis       : 0 = Liu の既定(VL), 1 = Jones ベクトル基底, 2 = 正規化法線ベクトル基底
 * less_memory : 1 なら中間行列を保存しない層ごとのソルバー（SolveInterior）を使う */
int s4l_set_polarization_options(S4_Simulation *S, int resolution, int basis, int less_memory){
	if(resolution >= 0){
		if(resolution < 2){ return -2; }
		S->options.resolution = resolution;
	}
	if(basis >= 0){
		S->options.use_jones_vector_basis = (1 == basis);
		S->options.use_normal_vector_basis = (2 == basis);
	}
	if(less_memory >= 0){
		S->options.use_less_memory = less_memory ? 1 : 0;
	}
	S4_Simulation_DestroyLayerModes(S, -1);
	Simulation_DestroySolution(S);
	Simulation_InvalidateFieldCache(S);
	return 0;
}

int s4l_set_verbosity(S4_Simulation *S, int verbosity){
	S->options.verbosity = verbosity;
	return 0;
}
