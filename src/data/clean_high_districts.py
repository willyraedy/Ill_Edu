def clean_data(raw, remove_outliers=True):
    for_model = raw.drop(columns=[
        'principal_turnover_within_6_years',
        'inst_and_support',
        'pupil_teacher_ratio__high_school',
        'district_type'
    ])
    for_model = for_model.dropna(subset=['%_ebf_capacity_to_meet_expectations'])

    if remove_outliers:
        for_model = for_model[for_model.teacher_attendace_rate > 5]
        for_model = for_model[for_model['%_student_enrollment__homeless'] < 30]

    for_model.columns = [x.replace('#', 'num').replace('%', 'perc') for x in for_model.columns]

    return for_model.reset_index().drop(columns=['index'])
