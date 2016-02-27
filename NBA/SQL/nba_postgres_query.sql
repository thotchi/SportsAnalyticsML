SELECT

box_home.game_id
, box_home.date_game
, box_home.team_city home_team
, box_away.team_city away_team
, box_home.pts home_pts
, box_away.pts away_pts
, box_home.pts - box_away.pts pts_diff

, box_home.pts_moving_avg::numeric(5,2) home_pts_moving_avg
, box_home.ptsa_moving_avg::numeric(5,2) home_ptsa_moving_avg
, box_home.pt_diff_moving_avg::numeric(5,2) home_pt_diff_moving_avg
, box_away.pts_moving_avg::numeric(5,2) away_pts_moving_avg
, box_away.ptsa_moving_avg::numeric(5,2) away_ptsa_moving_avg
, box_away.pt_diff_moving_avg::numeric(5,2) away_pt_diff_moving_avg

, box_home.turnovers_moving_avg::numeric(5,2) home_turnovers_moving_avg
, box_home.total_minutes_moving_avg::numeric(5,2) home_total_minutes_moving_avg
, box_home.ast_moving_avg::numeric(5,2) home_ast_moving_avg
, box_home.blk_moving_avg::numeric(5,2) home_blk_moving_avg
, box_home.dreb_moving_avg::numeric(5,2) home_dreb_moving_avg
, box_home.oreb_moving_avg::numeric(5,2) home_oreb_moving_avg
, box_home.reb_moving_avg::numeric(5,2) home_reb_moving_avg
, box_home.stl_moving_avg::numeric(5,2) home_stl_moving_avg
, box_home.fg3a_moving_avg::numeric(5,2) home_fg3a_moving_avg
, box_home.fg3m_moving_avg::numeric(5,2) home_fg3m_moving_avg
, box_home.fga_moving_avg::numeric(5,2) home_fga_moving_avg
, box_home.fgm_moving_avg::numeric(5,2) home_fgm_moving_avg
, box_home.fta_moving_avg::numeric(5,2) home_fta_moving_avg
, box_home.ftm_moving_avg::numeric(5,2) home_ftm_moving_avg
, box_home.pf_moving_avg::numeric(5,2) home_pf_moving_avg

, box_away.turnovers_moving_avg::numeric(5,2) away_turnovers_moving_avg
, box_away.total_minutes_moving_avg::numeric(5,2) away_total_minutes_moving_avg
, box_away.ast_moving_avg::numeric(5,2) away_ast_moving_avg
, box_away.blk_moving_avg::numeric(5,2) away_blk_moving_avg
, box_away.dreb_moving_avg::numeric(5,2) away_dreb_moving_avg
, box_away.oreb_moving_avg::numeric(5,2) away_oreb_moving_avg
, box_away.reb_moving_avg::numeric(5,2) away_reb_moving_avg
, box_away.stl_moving_avg::numeric(5,2) away_stl_moving_avg
, box_away.fg3a_moving_avg::numeric(5,2) away_fg3a_moving_avg
, box_away.fg3m_moving_avg::numeric(5,2) away_fg3m_moving_avg
, box_away.fga_moving_avg::numeric(5,2) away_fga_moving_avg
, box_away.fgm_moving_avg::numeric(5,2) away_fgm_moving_avg
, box_away.fta_moving_avg::numeric(5,2) away_fta_moving_avg
, box_away.ftm_moving_avg::numeric(5,2) away_ftm_moving_avg
, box_away.pf_moving_avg::numeric(5,2) away_pf_moving_avg

FROM
(
select 
game_id
, team_id
, date_game
, team_city
, home_away
, pts
, avg(turnovers) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) turnovers_moving_avg
, avg(total_minutes) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) total_minutes_moving_avg
, avg(ast) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ast_moving_avg
, avg(blk) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) blk_moving_avg
, avg(dreb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) dreb_moving_avg
, avg(oreb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) oreb_moving_avg
, avg(reb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) reb_moving_avg
, avg(stl) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) stl_moving_avg
, avg(pts) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pts_moving_avg
, avg(ptsa) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ptsa_moving_avg
, avg(pt_diff) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pt_diff_moving_avg
, avg(fg3a) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fg3a_moving_avg
, avg(fg3m) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fg3m_moving_avg
, avg(fga) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fga_moving_avg
, avg(fgm) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fgm_moving_avg
, avg(fta) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fta_moving_avg
, avg(ftm) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ftm_moving_avg
, avg(pf) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pf_moving_avg
FROM
box_master
) box_home
INNER JOIN
(
select 
game_id
, team_id
, date_game
, team_city
, home_away
, pts
, avg(turnovers) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) turnovers_moving_avg
, avg(total_minutes) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) total_minutes_moving_avg
, avg(ast) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ast_moving_avg
, avg(blk) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) blk_moving_avg
, avg(dreb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) dreb_moving_avg
, avg(oreb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) oreb_moving_avg
, avg(reb) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) reb_moving_avg
, avg(stl) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) stl_moving_avg
, avg(pts) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pts_moving_avg
, avg(ptsa) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ptsa_moving_avg
, avg(pt_diff) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pt_diff_moving_avg
, avg(fg3a) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fg3a_moving_avg
, avg(fg3m) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fg3m_moving_avg
, avg(fga) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fga_moving_avg
, avg(fgm) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fgm_moving_avg
, avg(fta) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) fta_moving_avg
, avg(ftm) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) ftm_moving_avg
, avg(pf) OVER (PARTITION BY team_id ORDER BY team_game_rank ASC ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) pf_moving_avg
FROM
box_master
) box_away
ON box_home.game_id = box_away.game_id
AND box_home.home_away = 'HOME'
AND box_away.home_away = 'AWAY'
ORDER BY box_home.date_game desc, box_home.game_id desc

limit 1000


