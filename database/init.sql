-- ============================================
-- AI技术文档助手系统 - 数据库初始化脚本
-- 适用版本: MySQL 8.0+
-- 使用说明: 在Navicat中打开并运行此脚本
-- ============================================

-- 1. 创建数据库
CREATE DATABASE IF NOT EXISTS tech_doc_assistant 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE tech_doc_assistant;

-- 2. 文档表
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    filename VARCHAR(255) NOT NULL COMMENT '存储的文件名',
    original_name VARCHAR(255) NOT NULL COMMENT '原始文件名',
    content_preview TEXT COMMENT '内容预览',
    chunk_count INT DEFAULT 0 COMMENT '切片数量',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='已上传的技术文档';

-- 3. 对话历史表
DROP TABLE IF EXISTS conversations;
CREATE TABLE conversations (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID（关联同一轮对话）',
    role VARCHAR(10) NOT NULL COMMENT '角色：user/assistant',
    content TEXT NOT NULL COMMENT '对话内容',
    related_chunks TEXT COMMENT '相关文档片段来源',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '消息时间',
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户与AI的对话记录';

-- 4. 缓存问答表
DROP TABLE IF EXISTS cache_entries;
CREATE TABLE cache_entries (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    question_hash VARCHAR(64) NOT NULL UNIQUE COMMENT '问题的MD5哈希值',
    question VARCHAR(1000) NOT NULL COMMENT '原始问题',
    answer TEXT NOT NULL COMMENT '回答内容',
    source_docs VARCHAR(500) COMMENT '来源文档',
    hit_count INT DEFAULT 0 COMMENT '命中次数',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_question_hash (question_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='高频问答对缓存';

-- 5. 验证创建结果
SELECT '数据库和表创建完成！' AS status;
SHOW TABLES;
