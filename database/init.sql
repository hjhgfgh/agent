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
-- 文档分两类，靠 is_builtin 区分生命周期：
--   内置文档（1）：随镜像发布，常驻知识库，接口层拒绝删除
--   临时文档（0）：访客上传，关掉网页后下次打开页面时被自动回收
-- 注：这两列是后加的。init.sql 只在数据卷首次初始化时执行一次，
--     因此老库升级靠后端启动时的 models._ensure_columns() 自动补列，
--     而不是重跑本脚本（那样会把已有数据 DROP 掉）。
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    filename VARCHAR(255) NOT NULL COMMENT '存储的文件名',
    original_name VARCHAR(255) NOT NULL COMMENT '原始文件名',
    content_preview TEXT COMMENT '内容预览',
    chunk_count INT DEFAULT 0 COMMENT '切片数量',
    is_builtin TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否内置文档：1=随镜像发布、常驻且不可删除',
    client_id VARCHAR(64) NULL COMMENT '上传者的浏览器会话标识，内置文档为 NULL',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    INDEX idx_client_id (client_id),
    INDEX idx_is_builtin (is_builtin)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库文档（内置 + 访客临时上传）';

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
