const testData =[
    {
        "id": "001",
        "instruction": "请用悲伤、失落的语气读出这句话。", // 对应 C-MOS 的 Context
        "text": "也许我们真的到了该结束的时候了。", // 文本内容
        "audios": {
            "Baseline_1": "./audio/001_b1.wav",
            "Baseline_2": "./audio/001_b2.wav",
            "Baseline_3": "./audio/001_b3.wav",
            "Baseline_4": "./audio/001_b4.wav",
            "My_Model": "./audio/001_my.wav"
        }
    },
    {
        "id": "002",
        "instruction": "请用极其兴奋和激动的语气朗读。",
        "text": "我终于拿到了梦寐以求的录取通知书！",
        "audios": {
            "Baseline_1": "./audio/002_b1.wav",
            "Baseline_2": "./audio/002_b2.wav",
            "Baseline_3": "./audio/002_b3.wav",
            "Baseline_4": "./audio/002_b4.wav",
            "My_Model": "./audio/002_my.wav"
        }
    }
    // 继续添加你的数据...
];