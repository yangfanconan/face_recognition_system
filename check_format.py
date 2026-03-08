with open('datasets/widerface/wider_face_split/wider_face_train_bbx_gt.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print(f"Total lines: {len(lines)}")
print("\nFirst 50 lines:")
for i in range(min(50, len(lines))):
    line = lines[i].strip()
    print(f"{i}: {line[:100]}")
