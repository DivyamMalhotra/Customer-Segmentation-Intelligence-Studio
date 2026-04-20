"""
Replace K-Means tab2 content (lines 1370-1398) with the three-chart
cluster visualisation matching the reference screenshot.
"""
import pathlib

APP = pathlib.Path(r'e:\Supervised Project\app\app.py')
lines = APP.read_text(encoding='utf-8').splitlines(keepends=True)

# The new tab2 block (replaces lines 1370-1398, 0-indexed: 1369-1397)
NEW_TAB2 = '''\
        with tab2:
            km_labels, km_sil, km_db, km_model = run_kmeans(features_scaled, k_clusters)
            scored['KM_Cluster'] = km_labels
            scored['KM_Label']   = scored['KM_Cluster'].apply(lambda x: f'Cluster {x}')

            # ── Summary table + pie ────────────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                cs = scored.groupby('KM_Label').agg(
                    Count        = ('CustomerID',         'count'),
                    Avg_Age      = ('Age',                'mean'),
                    Avg_Income   = ('Annual Income (k$)', 'mean'),
                    Avg_Spending = ('Spending Score',     'mean'),
                ).round(1).reset_index()
                st.dataframe(cs, use_container_width=True)
            with c2:
                fig = px.pie(cs, names='KM_Label', values='Count',
                             color_discrete_sequence=CLUSTER_PALETTE, hole=0.4)
                dark_layout(fig, 'Cluster Sizes')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""<div class='info-box'>
            Silhouette: <strong>{km_sil:.3f}</strong> &nbsp;|&nbsp;
            Davies-Bouldin: <strong>{km_db:.3f}</strong> &nbsp;|&nbsp;
            k = <strong>{k_clusters}</strong>
            </div>""", unsafe_allow_html=True)

            # ── Two 2-D feature-axis scatter plots ─────────────────
            st.markdown("<div class=\'section-header\'>📍 Cluster Visualizations</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig_inc_sp = px.scatter(
                    scored, x='Annual Income (k$)', y='Spending Score',
                    color='KM_Label',
                    color_discrete_sequence=CLUSTER_PALETTE,
                    opacity=0.82,
                    labels={'KM_Label': 'Cluster',
                            'Spending Score': 'Spending Score (1-100)'},
                    hover_data=['CustomerID', 'Gender', 'Age']
                )
                dark_layout(fig_inc_sp, 'Annual Income vs Spending Score', height=400)
                fig_inc_sp.update_traces(
                    marker=dict(size=7, line=dict(width=0.4,
                                color='rgba(0,0,0,0.3)')))
                st.plotly_chart(fig_inc_sp, use_container_width=True)

            with c2:
                fig_age_inc = px.scatter(
                    scored, x='Age', y='Annual Income (k$)',
                    color='KM_Label',
                    color_discrete_sequence=CLUSTER_PALETTE,
                    opacity=0.82,
                    labels={'KM_Label': 'Cluster'},
                    hover_data=['CustomerID', 'Gender', 'Spending Score']
                )
                dark_layout(fig_age_inc, 'Age vs Annual Income', height=400)
                fig_age_inc.update_traces(
                    marker=dict(size=7, line=dict(width=0.4,
                                color='rgba(0,0,0,0.3)')))
                st.plotly_chart(fig_age_inc, use_container_width=True)

            # ── Full 3-D cluster view ──────────────────────────────
            fig_3d = px.scatter_3d(
                scored, x='Age', y='Annual Income (k$)', z='Spending Score',
                color='KM_Label',
                color_discrete_sequence=CLUSTER_PALETTE,
                opacity=0.78,
                labels={'KM_Label': 'Cluster'},
                hover_data=['CustomerID', 'Gender']
            )
            fig_3d.update_traces(marker=dict(size=4))
            fig_3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                title='3D Cluster View \u2014 Age / Annual Income / Spending Score',
                title_font=dict(family='Syne', size=14, color='#90cdf4'),
                scene=dict(
                    xaxis=dict(title='Age',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    yaxis=dict(title='Annual Income (k$)',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    zaxis=dict(title='Spending Score',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    bgcolor='rgba(8,12,20,0.95)',
                ),
                height=570,
                margin=dict(t=50, b=5, l=5, r=5),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
            )
            st.plotly_chart(fig_3d, use_container_width=True)

'''

# Lines are 1-indexed; 1370-1398 => 0-indexed 1369 to 1397 (inclusive)
START = 1369   # 0-indexed, line 1370
END   = 1398   # 0-indexed, line 1398 (exclusive – next line after old block)

before = lines[:START]
after  = lines[END:]

new_lines = before + [NEW_TAB2] + after
APP.write_text(''.join(new_lines), encoding='utf-8')
print(f"Patched: replaced lines 1370-1398. New total lines: {len(new_lines)}")
