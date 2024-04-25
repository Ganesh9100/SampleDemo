def check_null_val(table,tb):
        count = 0
        total = 0
        for val in table[tb][0]:
            if val == ' ':
                count = count + 1
            total = total + 1
        percent_ = (count//total) * 100 
        return percent_
        
    def get_table_key_pair(table,tn):
        keys = table[tn][0]
        final = []
        for i in range(len(table[tn])):
            dummy = {}
            if i!=0:
                for j in range(len(table[tn][0])):
                    dummy[f'{keys[j]}'] = table[tn][i][j]
                final.append(dummy)
        return final
    
    
    def post_process_table(a):
        final_report = {}
        table_names = list(a.keys())
  # print(table_names)
        for tn in table_names:
            try:
                if len(a[tn][0])==2:
                    print("going to 1st if")
                    #final_report[tn]=[a[tn]]
                else:
                    if check_null_val(a,tn) > 20:
                        print("going to 2nd if")
                        a[tn].pop(0)
                        final_report[tn] = get_table_key_pair(a,tn)
                    else:
                        print("going to 3st elsee")
                        final_report[tn] = get_table_key_pair(a,tn)
            except Exception as E:
                    print("e",E)
                    final_report[tn] =a[tn]
        return final_report

    
    def remove_form_duplicates(a):
        pages = list(a.keys())
        new_form = {"Forms" : {}}
        new_table = {"Tables" : {}}
        for page in pages:
            for k,v in a[page].items():
                if k[:5]== "table":
                    new_table["Tables"][k]=v
                else:
                    new_form["Forms"][k]=v
        # blank = {"Divider" : "*******"}
        result_rem = {**new_form,**new_table}
        return result_rem
