import os
import json
import base64
import io
from flask import Flask, send_from_directory, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

app = Flask(__name__)
DATA_FILE = 'planner_data.json'

# --- Serve Frontend ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- Data API ---
@app.route('/api/data', methods=['GET'])
def get_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/data', methods=['POST'])
def save_data():
    data = request.json
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "success"})

# --- Statistics API ---
@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Keine Daten vorhanden. Benutze zuerst den Planer!"})

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    state = data.get('state', {})
    cats = data.get('cats', [])
    tasks = data.get('tasks', [])
    study_goals = data.get('studyGoals', {})

    if not state or not cats or not tasks:
        return jsonify({"error": "Nicht genug Daten für Statistiken vorhanden."})

    cat_map = {c['id']: c['name'] for c in cats}
    cat_colors_map = {c['id']: c.get('color', 'green') for c in cats}

    COLOR_MAP = {
        'green': '#1D9E75', 'blue': '#185FA5', 'amber': '#D4890B',
        'pink': '#C04670', 'teal': '#0E8064', 'purple': '#534AB7',
        'red': '#E24B4A', 'coral': '#D85A30',
    }

    # ========== 1. Category Completion Rates ==========
    cat_totals = {c['id']: 0 for c in cats}
    cat_dones = {c['id']: 0 for c in cats}

    for wk, week_data in state.items():
        for i in range(7):
            for t in tasks:
                cat_id = t['cat']
                if cat_id not in cat_totals:
                    continue
                cat_totals[cat_id] += 1
                key = f"{i}_{t['id']}"
                if week_data.get(key):
                    cat_dones[cat_id] += 1

    completion_rates = []
    for cid in cat_totals:
        total = cat_totals[cid]
        if total > 0:
            rate = (cat_dones[cid] / total) * 100
        else:
            rate = 0
        completion_rates.append({
            'cat_id': cid,
            'Category': cat_map.get(cid, cid),
            'Rate': round(rate, 1),
            'Color': COLOR_MAP.get(cat_colors_map.get(cid, 'green'), '#1D9E75')
        })

    df_rates = pd.DataFrame(completion_rates)

    # ========== 2. Task-Level Analysis ==========
    task_totals = {t['id']: 0 for t in tasks}
    task_dones = {t['id']: 0 for t in tasks}

    for wk, week_data in state.items():
        for i in range(7):
            for t in tasks:
                task_totals[t['id']] += 1
                key = f"{i}_{t['id']}"
                if week_data.get(key):
                    task_dones[t['id']] += 1

    task_stats = []
    for t in tasks:
        total = task_totals[t['id']]
        done = task_dones[t['id']]
        rate = (done / total * 100) if total > 0 else 0
        task_stats.append({
            'Task': t['label'],
            'Category': cat_map.get(t['cat'], '?'),
            'Rate': round(rate, 1),
            'Done': done,
            'Total': total,
            'Color': COLOR_MAP.get(cat_colors_map.get(t['cat'], 'green'), '#1D9E75')
        })

    df_tasks = pd.DataFrame(task_stats).sort_values('Rate')

    # ========== 3. Time Tracking Analysis ==========
    cat_time = {c['id']: 0 for c in cats}
    for wk, week_data in state.items():
        for key, val in week_data.items():
            if '_time_' in key and isinstance(val, (int, float)):
                cat_id = key.split('_time_')[1]
                if cat_id in cat_time:
                    cat_time[cat_id] += val

    time_stats = []
    for cid, secs in cat_time.items():
        if secs > 0:
            time_stats.append({
                'Category': cat_map.get(cid, cid),
                'Minutes': round(secs / 60, 1),
                'Color': COLOR_MAP.get(cat_colors_map.get(cid, 'green'), '#1D9E75')
            })

    df_time = pd.DataFrame(time_stats)

    # ========== 4. Words Learned Count ==========
    total_words = 0
    for wk, week_data in state.items():
        for key, val in week_data.items():
            if key.endswith('_words') and isinstance(val, list):
                total_words += len(val)

    # ========== GENERATE CHARTS ==========
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('#f5f5f0')
    fig.suptitle('Your Learning Statistics', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Completion rate by category
    ax1 = axes[0, 0]
    if not df_rates.empty:
        bars = ax1.barh(df_rates['Category'], df_rates['Rate'], color=df_rates['Color'], edgecolor='white', height=0.6)
        ax1.set_xlim(0, 100)
        ax1.set_title('Completion Rate by Category', fontsize=11, fontweight='bold')
        ax1.set_xlabel('%')
        for bar, rate in zip(bars, df_rates['Rate']):
            ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                     f'{rate:.0f}%', va='center', fontsize=9, fontweight='bold')
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_facecolor('#f5f5f0')

    # Chart 2: Most skipped tasks (bottom 5)
    ax2 = axes[0, 1]
    if not df_tasks.empty:
        bottom5 = df_tasks.head(5)
        labels = [l[:25] + '...' if len(l) > 25 else l for l in bottom5['Task']]
        bars = ax2.barh(labels, bottom5['Rate'], color=bottom5['Color'], edgecolor='white', height=0.6)
        ax2.set_xlim(0, 100)
        ax2.set_title('Most Skipped Tasks', fontsize=11, fontweight='bold')
        ax2.set_xlabel('%')
        for bar, rate in zip(bars, bottom5['Rate'].values):
            ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                     f'{rate:.0f}%', va='center', fontsize=9, fontweight='bold')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_facecolor('#f5f5f0')

    # Chart 3: Time spent per category
    ax3 = axes[1, 0]
    if not df_time.empty:
        bars = ax3.barh(df_time['Category'], df_time['Minutes'], color=df_time['Color'], edgecolor='white', height=0.6)
        ax3.set_title('Learning Time by Category', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Minutes')
        for bar, mins in zip(bars, df_time['Minutes']):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{mins:.0f}m', va='center', fontsize=9, fontweight='bold')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No Timer Used', ha='center', va='center', transform=ax3.transAxes, color='gray')
    ax3.set_facecolor('#f5f5f0')

    # Chart 4: Summary text panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_facecolor('#f5f5f0')
    num_weeks = len(state)
    total_tasks_done = sum(task_dones.values())
    total_tasks_all = sum(task_totals.values())
    overall_rate = (total_tasks_done / total_tasks_all * 100) if total_tasks_all > 0 else 0
    total_time_min = sum(cat_time.values()) / 60

    summary_lines = [
        f"📅  Weeks Tracked: {num_weeks}",
        f"✅  Tasks Completed: {total_tasks_done}/{total_tasks_all} ({overall_rate:.0f}%)",
        f"⏱️  Total Learning Time: {total_time_min:.0f} Minutes",
        f"📖  Words Learned: {total_words}",
    ]
    y_pos = 0.85
    ax4.text(0.05, 0.95, 'Summary', fontsize=12, fontweight='bold',
             transform=ax4.transAxes, va='top')
    for line in summary_lines:
        ax4.text(0.05, y_pos, line, fontsize=11, transform=ax4.transAxes, va='top', linespacing=1.5)
        y_pos -= 0.18

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # ========== WEAKNESS ANALYSIS ==========
    weaknesses = []
    if not df_rates.empty:
        df_sorted = df_rates.sort_values('Rate')
        worst = df_sorted.iloc[0]
        if worst['Rate'] < 50:
            weaknesses.append(
                f"Your biggest weakness is <b>{worst['Category']}</b> — "
                f"only <b>{worst['Rate']:.0f}%</b> completion rate. "
                f"Try to complete more tasks here!"
            )

    if not df_tasks.empty:
        worst_task = df_tasks.iloc[0]
        if worst_task['Rate'] < 30:
            task_name = worst_task['Task']
            task_rate = worst_task['Rate']
            weaknesses.append(
                f'The task <b>{task_name}</b> is skipped most often '
                f'(<b>{task_rate:.0f}%</b>). Make it a priority!'
            )

    # Time vs goals
    for cid, goal_min in study_goals.items():
        if isinstance(goal_min, (int, float)) and goal_min > 0:
            actual_min = cat_time.get(cid, 0) / 60
            expected = goal_min * 7 * num_weeks  # goal per day * 7 days * weeks
            if expected > 0 and actual_min < expected * 0.5:
                cat_name = cat_map.get(cid, cid)
                weaknesses.append(
                    f"You are not reaching your time goal for <b>{cat_name}</b> — "
                    f"<b>{actual_min:.0f}m</b> out of <b>{expected:.0f}m</b> goal. "
                    f"Try to schedule more time!"
                )

    if not weaknesses:
        weaknesses.append("No weaknesses detected — you are doing great! 🎉 Keep it up!")

    return jsonify({
        "chart_base64": chart_base64,
        "weaknesses": weaknesses
    })


if __name__ == '__main__':
    print("\n[+] Deutsch Wochenplaner laeuft auf: http://localhost:5000\n")
    app.run(port=5000, debug=True)
